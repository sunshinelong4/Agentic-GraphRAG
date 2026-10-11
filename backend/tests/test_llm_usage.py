"""LLM 用量与成本追踪的单元测试

为什么这个模块值得单独测：
成本与延迟是 Agent 工程的核心指标，而这类指标最容易"悄悄错"——
字段名写错、估算值被当成真实值、并发请求互相污染，都不会让程序报错，
只会让你在答辩时给出一个错误数字。所以这里把口径逐条钉死。
"""

import pytest

from app.services.llm_usage import (
    LLMCallRecord,
    LLMUsageTracker,
    estimate_tokens,
    extract_usage,
)


# =============================================================================
# token 估算
# =============================================================================


class TestEstimateTokens:
    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_chinese_denser_than_english(self):
        """同样字符数下中文的 token 数应明显高于英文（这是估算公式的核心假设）。"""
        chinese = estimate_tokens("张建国患有扩张型心肌病")
        english = estimate_tokens("abcdefghijkl")
        assert chinese > english

    def test_english_roughly_quarter(self):
        """英文按 4 字符约 1 token。"""
        assert estimate_tokens("a" * 40) == pytest.approx(11, abs=2)

    def test_scales_with_length(self):
        assert estimate_tokens("张建国" * 10) > estimate_tokens("张建国")

    def test_never_returns_negative(self):
        assert estimate_tokens(" ") >= 0


# =============================================================================
# usage 抽取（两种字段格式）
# =============================================================================


class _Resp:
    """模拟 LangChain 返回值。"""

    def __init__(self, content="", usage_metadata=None, response_metadata=None):
        self.content = content
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        if response_metadata is not None:
            self.response_metadata = response_metadata


class TestExtractUsage:
    def test_langchain_standard_field(self):
        """LangChain 标准字段 usage_metadata 优先。"""
        resp = _Resp(usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165})
        prompt, completion, estimated = extract_usage(resp)
        assert (prompt, completion, estimated) == (120, 45, False)

    def test_openai_native_field(self):
        """没有 usage_metadata 时退回 OpenAI 原生字段。"""
        resp = _Resp(response_metadata={"token_usage": {"prompt_tokens": 80, "completion_tokens": 20}})
        prompt, completion, estimated = extract_usage(resp)
        assert (prompt, completion, estimated) == (80, 20, False)

    def test_langchain_field_wins_over_openai_field(self):
        resp = _Resp(
            usage_metadata={"input_tokens": 100, "output_tokens": 10},
            response_metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )
        assert extract_usage(resp)[:2] == (100, 10)

    def test_falls_back_to_estimate(self):
        prompt, completion, estimated = extract_usage(_Resp(content="答案"), "系统提示", "生成的答案文本")
        assert estimated is True
        assert prompt > 0 and completion > 0

    def test_empty_usage_dict_falls_back_to_estimate(self):
        """部分网关会返回空的 usage 对象；不能把它当成"0 token"。"""
        resp = _Resp(usage_metadata={}, response_metadata={})
        _, _, estimated = extract_usage(resp, "提示", "答案")
        assert estimated is True

    def test_zero_usage_falls_back_to_estimate(self):
        """usage 全是 0 说明字段没填，同样要退回估算。"""
        resp = _Resp(usage_metadata={"input_tokens": 0, "output_tokens": 0})
        assert extract_usage(resp, "提示", "答案")[2] is True

    def test_non_dict_metadata_is_safe(self):
        resp = _Resp(usage_metadata="坏数据", response_metadata=None)
        assert extract_usage(resp, "提示", "答案")[2] is True


# =============================================================================
# 单条记录
# =============================================================================


class TestCallRecord:
    def test_total_tokens(self):
        record = LLMCallRecord("generate", "m", 100, 50, 1200.0)
        assert record.total_tokens == 150

    def test_to_dict_includes_total(self):
        payload = LLMCallRecord("plan", "m", 10, 5, 100.0).to_dict()
        assert payload["total_tokens"] == 15
        assert payload["node"] == "plan"


# =============================================================================
# 追踪器
# =============================================================================


class TestTracker:
    def test_empty_summary(self):
        summary = LLMUsageTracker(model="m").summary()
        assert summary["llm_calls"] == 0
        assert summary["total_tokens"] == 0
        assert summary["cost"] is None
        assert summary["cost_configured"] is False

    def test_accumulates_calls(self):
        tracker = LLMUsageTracker(model="m")
        tracker.record("plan", 100, 20, 300.0)
        tracker.record("generate", 500, 200, 1500.0)

        summary = tracker.summary()
        assert summary["llm_calls"] == 2
        assert summary["prompt_tokens"] == 600
        assert summary["completion_tokens"] == 220
        assert summary["total_tokens"] == 820
        assert summary["llm_latency_ms"] == 1800.0

    def test_by_node_attribution(self):
        """按节点归因是优化的前提：不区分节点就不知道钱花在哪一步。"""
        tracker = LLMUsageTracker(model="m")
        tracker.record("plan", 100, 10, 200.0)
        tracker.record("plan", 100, 10, 200.0)
        tracker.record("generate", 900, 300, 3000.0)

        by_node = tracker.summary()["by_node"]
        assert by_node["plan"]["calls"] == 2
        assert by_node["plan"]["prompt_tokens"] == 200
        assert by_node["generate"]["calls"] == 1
        assert by_node["generate"]["prompt_tokens"] == 900

    def test_estimated_calls_counted(self):
        tracker = LLMUsageTracker(model="m")
        tracker.record("generate", 10, 5, 100.0, estimated=True)
        tracker.record("plan", 10, 5, 100.0, estimated=False)
        summary = tracker.summary()
        assert summary["estimated_calls"] == 1

    def test_failed_calls_counted(self):
        """失败请求的成本不能被静默吞掉。"""
        tracker = LLMUsageTracker(model="m")
        tracker.record("route", 50, 0, 80.0, estimated=True, error="连接超时")
        summary = tracker.summary()
        assert summary["failed_calls"] == 1
        assert summary["prompt_tokens"] == 50

    def test_record_from_response_uses_real_usage(self):
        tracker = LLMUsageTracker(model="m")
        resp = _Resp(usage_metadata={"input_tokens": 200, "output_tokens": 80})
        tracker.record_from_response("generate", resp, "提示" * 100, "答案", 900.0)

        summary = tracker.summary()
        assert summary["prompt_tokens"] == 200
        assert summary["estimated_calls"] == 0

    def test_record_stream_estimate_marked(self):
        """流式响应不返回 usage，必须标记为估算值。"""
        tracker = LLMUsageTracker(model="m")
        tracker.record_stream_estimate("generate", "很长很长的提示词", "答案文本", 500.0)

        summary = tracker.summary()
        assert summary["estimated_calls"] == 1
        assert summary["completion_tokens"] > 0

    def test_model_taken_from_response_metadata(self):
        tracker = LLMUsageTracker(model="default-model")
        resp = _Resp(
            usage_metadata={"input_tokens": 1, "output_tokens": 1},
            response_metadata={"model_name": "actual-model"},
        )
        tracker.record_from_response("generate", resp, "p", "c", 10.0)
        assert tracker.summary()["calls"][0]["model"] == "actual-model"


class TestCost:
    def test_cost_none_when_price_not_configured(self):
        """没配置单价时给 None，而不是给 0 —— 0 会被误读成"免费"。"""
        tracker = LLMUsageTracker(model="m")
        tracker.record("generate", 1_000_000, 0, 100.0)
        assert tracker.cost is None
        assert tracker.summary()["cost"] is None

    def test_cost_computed_when_configured(self):
        tracker = LLMUsageTracker(
            model="m", price_input_per_1m=2.0, price_output_per_1m=8.0
        )
        tracker.record("generate", 1_000_000, 500_000, 100.0)
        # 1M * 2 + 0.5M * 8 = 2 + 4 = 6
        assert tracker.cost == pytest.approx(6.0)

    def test_cost_partial_configuration(self):
        """只配了输入单价也应能算出费用（输出按 0 计）。"""
        tracker = LLMUsageTracker(model="m", price_input_per_1m=1.0)
        tracker.record("generate", 2_000_000, 100, 100.0)
        assert tracker.cost == pytest.approx(2.0)
        assert tracker.cost_configured is True

    def test_cost_accumulates_across_calls(self):
        tracker = LLMUsageTracker(model="m", price_input_per_1m=1.0, price_output_per_1m=1.0)
        tracker.record("plan", 100_000, 0, 10.0)
        tracker.record("generate", 200_000, 0, 10.0)
        assert tracker.cost == pytest.approx(0.3)


class TestIsolation:
    def test_two_trackers_do_not_mix(self):
        """两个请求各自的追踪器必须完全隔离（并发场景的正确性前提）。"""
        first = LLMUsageTracker(model="m")
        second = LLMUsageTracker(model="m")

        first.record("generate", 100, 50, 100.0)
        second.record("plan", 1, 1, 10.0)

        assert first.summary()["total_tokens"] == 150
        assert second.summary()["total_tokens"] == 2
        assert first.summary()["llm_calls"] == 1
