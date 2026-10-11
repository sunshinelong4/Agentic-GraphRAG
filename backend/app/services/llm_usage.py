"""LLM 调用与成本追踪

为什么需要单独一个模块
----------------------
Agent 每回答一个问题会发起**多次** LLM 调用（路由判官、充分性判官、查询改写、答案生成、
必要时还有关系抽取）。不做计量的话，三个问题讲不清：

1. **成本**：这个 Agent 每问一次花多少钱？加了多轮工具调用后涨了多少？
2. **性能归因**：延迟到底耗在哪一步？（本项目的实测结论：生成占绝大部分，检索可忽略）
3. **回归检测**：改了提示词之后 token 是不是悄悄翻倍了？

设计取舍
--------
**优先用 API 返回的真实 usage，拿不到才估算。**
`ChatOpenAI` 的返回通常带 `usage_metadata`（LangChain 标准字段）或
`response_metadata["token_usage"]`（OpenAI 原生字段），两者格式不同，这里都兼容。
流式响应往往不带 usage，此时退回估算，并在记录里标 `estimated=True` ——
**估算值必须能被识别出来**，否则指标会悄悄失真。

**不硬编码价格。**
各家模型单价会变，写死在代码里迟早变成错误信息。价格通过配置传入，
未配置时只报 token、`cost` 返回 `None`，并在结果里标明 `cost_configured=False`。
宁可少一个数字，也不要一个假数字。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# token 估算
# =============================================================================


def _is_cjk(ch: str) -> bool:
    """判断是否为中日韩表意文字（这类字符的 token 密度与英文差很多）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一表意文字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3000 <= code <= 0x303F   # CJK 标点
        or 0xFF00 <= code <= 0xFFEF   # 全角字符
    )


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    刻意**不引入 tiktoken**，两个原因：
    1. `tiktoken.get_encoding()` 首次调用要联网下载 BPE 词表，在离线/沙箱环境会直接失败；
    2. tiktoken 是 OpenAI 的切分方式，对 DeepSeek 本来就不是精确的。

    既然都不精确，就用一个简单、透明、无依赖的近似：
        中文（CJK）按 1 字符 ≈ 1 token，其余（英文/数字/空格）按 4 字符 ≈ 1 token。

    这个估算只用于"API 没返回 usage"的兜底场景，且记录里会标 `estimated=True`。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


# =============================================================================
# 单次调用记录
# =============================================================================


@dataclass
class LLMCallRecord:
    """一次 LLM 调用的计量记录。"""

    node: str                 # 发起调用的节点（plan / assess / generate / rewrite / extract …）
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    estimated: bool = False   # token 数是否为估算值（True 表示 API 没返回 usage）
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


# =============================================================================
# 追踪器
# =============================================================================


class LLMUsageTracker:
    """一次请求（或一次评测）内的 LLM 用量累加器。

    生命周期与"一次问答"绑定：在 `answer()` / `answer_stream()` 入口创建，
    跟随 state 传给各节点。这样并发请求之间互不干扰 ——
    如果做成 Agent 上的成员变量，FastAPI 线程池里两个并发请求会把用量混在一起。

    线程安全：单次请求内是顺序执行，因此不额外加锁；跨请求靠"每个请求一个实例"隔离。
    """

    def __init__(
        self,
        model: str = "",
        price_input_per_1m: float = 0.0,
        price_output_per_1m: float = 0.0,
    ) -> None:
        self.model = model
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.calls: List[LLMCallRecord] = []

    # -------------------------------------------------------------------------
    # 记录入口
    # -------------------------------------------------------------------------

    def record(
        self,
        node: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        estimated: bool = False,
        error: str = "",
        model: str = "",
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            node=node,
            model=model or self.model,
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
            latency_ms=round(latency_ms, 2),
            estimated=estimated,
            error=error,
        )
        self.calls.append(record)
        return record

    def record_from_response(
        self,
        node: str,
        response: Any,
        prompt_text: str,
        completion_text: str,
        latency_ms: float,
    ) -> LLMCallRecord:
        """从 LLM 返回值里抽取真实 usage；抽不到则估算。"""
        prompt_tokens, completion_tokens, estimated = extract_usage(
            response, prompt_text, completion_text
        )
        model = self.model
        metadata = getattr(response, "response_metadata", None) or {}
        if isinstance(metadata, dict):
            model = metadata.get("model_name") or metadata.get("model") or model
        return self.record(
            node=node,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            estimated=estimated,
            model=model,
        )

    def record_stream_estimate(
        self,
        node: str,
        prompt_text: str,
        completion_text: str,
        latency_ms: float,
        error: str = "",
    ) -> LLMCallRecord:
        """流式调用通常不返回 usage，按估算记录（并标记 estimated）。

        `error` 用来标记"流式中途失败"：失败请求的 token 消耗同样是真的，
        不记的话成本报表会偏低，而偏低只会让你在优化时做出错误判断。
        """
        return self.record(
            node=node,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(completion_text),
            latency_ms=latency_ms,
            estimated=True,
            error=error,
        )

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------

    @property
    def cost_configured(self) -> bool:
        return self.price_input_per_1m > 0 or self.price_output_per_1m > 0

    @property
    def cost(self) -> Optional[float]:
        """估算费用。未配置单价时返回 None —— 宁可没有数字，也不要假数字。"""
        if not self.cost_configured:
            return None
        prompt = sum(c.prompt_tokens for c in self.calls)
        completion = sum(c.completion_tokens for c in self.calls)
        return round(
            prompt / 1_000_000 * self.price_input_per_1m
            + completion / 1_000_000 * self.price_output_per_1m,
            6,
        )

    def summary(self) -> Dict[str, Any]:
        """汇总结果。这是会写进 API 响应与评测报告的内容。"""
        by_node: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        )
        for call in self.calls:
            bucket = by_node[call.node]
            bucket["calls"] += 1
            bucket["prompt_tokens"] += call.prompt_tokens
            bucket["completion_tokens"] += call.completion_tokens
            bucket["latency_ms"] = round(bucket["latency_ms"] + call.latency_ms, 2)

        return {
            "model": self.model,
            "llm_calls": len(self.calls),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "total_tokens": sum(c.total_tokens for c in self.calls),
            "llm_latency_ms": round(sum(c.latency_ms for c in self.calls), 2),
            "estimated_calls": sum(1 for c in self.calls if c.estimated),
            "failed_calls": sum(1 for c in self.calls if c.error),
            "cost": self.cost,
            "cost_configured": self.cost_configured,
            "by_node": {node: dict(bucket) for node, bucket in sorted(by_node.items())},
            "calls": [c.to_dict() for c in self.calls],
        }


# =============================================================================
# usage 抽取（兼容两种字段格式）
# =============================================================================


def extract_usage(
    response: Any, prompt_text: str = "", completion_text: str = ""
) -> Tuple[int, int, bool]:
    """从 LangChain 返回值里取 token 用量。

    要兼容两种来源：
    - `response.usage_metadata`（LangChain 标准）: {"input_tokens", "output_tokens"}
    - `response.response_metadata["token_usage"]`（OpenAI 原生）:
      {"prompt_tokens", "completion_tokens"}

    Returns:
        (prompt_tokens, completion_tokens, estimated)
        - 两个来源都拿不到时，用估算值并返回 estimated=True
        - 官方字段存在但为 0/None 时也退回估算（部分网关会返回空 usage）
    """
    # 来源 1：LangChain 标准字段
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, dict):
        prompt = metadata.get("input_tokens")
        completion = metadata.get("output_tokens")
        if isinstance(prompt, int) and isinstance(completion, int) and (prompt or completion):
            return prompt, completion, False

    # 来源 2：OpenAI 原生字段
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            prompt = token_usage.get("prompt_tokens")
            completion = token_usage.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt, completion, False

    # 兜底：估算
    return estimate_tokens(prompt_text), estimate_tokens(completion_text), True


class _Timer:
    """记录代码块耗时的小工具（配合 with 语句使用）。"""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.elapsed_ms = 0.0

    def __enter__(self) -> "_Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed_ms = round((time.perf_counter() - self.started) * 1000, 2)


def timer() -> _Timer:
    return _Timer()
