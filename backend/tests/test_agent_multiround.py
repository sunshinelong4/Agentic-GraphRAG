"""多轮工具调用与规划失败的测试

覆盖本次改造的三个新增能力：
1. **ReAct 式多轮工具循环**：`plan → tools → assess → plan`
2. **规划失败降级链**：非法工具名 / 空工具集 / 工具全空都有兜底
3. **证据累积**：第二轮不能覆盖第一轮（这是多轮改造最容易踩的坑）

全部离线、确定性：`use_llm_router=False` + `use_llm_assessor=False` 时整条链路不依赖模型，
只有显式打开判官的用例才脚本化 FakeLLM。
"""

import pytest

from app.services.agent_qa import (
    ALL_TOOLS,
    DEFAULT_PLAN_TOOLS,
    TOOL_GRAPH,
    TOOL_GRAPH_GLOBAL,
    TOOL_KB_META,
    TOOL_VECTOR,
    GraphRAGAgent,
    _digest_evidence,
    _parse_json_object,
)
from app.services.graph_store import KnowledgeGraphStore, RelationTriple
from tests.test_agent_qa import FakeLLM, FakeVectorStore


# =============================================================================
# 测试用图谱
# =============================================================================


@pytest.fixture
def graph_store():
    store = KnowledgeGraphStore()
    store.add_extractions(
        "d1", "病历.pdf",
        [
            {"extraction_class": "患者", "extraction_text": "张建国",
             "char_interval": {"start_pos": 0, "end_pos": 3}},
            {"extraction_class": "药物", "extraction_text": "沙库巴曲缬沙坦钠片",
             "char_interval": {"start_pos": 10, "end_pos": 20}},
            {"extraction_class": "机构", "extraction_text": "诺华",
             "char_interval": {"start_pos": 30, "end_pos": 32}},
        ],
    )
    store.add_triples(
        [
            RelationTriple("张建国", "服用", "沙库巴曲缬沙坦钠片", evidence="服用沙库巴曲缬沙坦钠片"),
            RelationTriple("沙库巴曲缬沙坦钠片", "研发", "诺华", evidence="诺华研发"),
        ],
        doc_id="d1", doc_title="病历.pdf", auto_create_entities=False,
    )
    return store


VECTOR_RESULTS = [
    {"score": 0.9, "chunk_id": "c1", "doc_id": "d1", "doc_title": "病历.pdf",
     "content": "张建国服用沙库巴曲缬沙坦钠片。", "chunk_type": "document",
     "attributes": {"paragraph_index": 0}},
]


def _agent(graph_store, vector_store, responses="答案。", **kwargs):
    kwargs.setdefault("use_llm_router", False)
    kwargs.setdefault("use_llm_assessor", False)
    return GraphRAGAgent(
        graph_store=graph_store,
        vector_store=vector_store,
        llm=FakeLLM(responses),
        **kwargs,
    )


def _tool_entries(result):
    return [e for e in result["trace"] if e["node"] == "tools"]


def _plan_entries(result):
    return [e for e in result["trace"] if e["node"] == "plan"]


def _assess_entries(result):
    return [e for e in result["trace"] if e["node"] == "assess"]


# =============================================================================
# 1. 多轮循环：检索为空时换工具补查
# =============================================================================


class TestMultiRoundSwitchOnEmptyEvidence:
    """最常见、也最有价值的续查场景：本轮什么都没检索到。

    这时盲目生成答案必然是一句"无法回答"，而换一个检索工具往往就能拿到东西。
    """

    def test_empty_evidence_triggers_second_round(self, graph_store):
        # 图谱里没有"完全不存在的实体"，图检索必然为空；向量库也返回空
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        result = agent.answer("完全不存在的实体名称")

        assert result["tool_rounds"] == 2, "第一轮全空时应换工具再查一轮"
        assessments = _assess_entries(result)
        assert assessments[0]["decision"] == "plan"
        assert assessments[0]["reason"] == "empty_evidence_switch_tools"

    def test_second_round_uses_unused_tool(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        result = agent.answer("完全不存在的实体名称")

        first_round_tools = _plan_entries(result)[0]["tools"]
        second_round_tools = _plan_entries(result)[1]["tools"]
        assert not set(second_round_tools) & set(first_round_tools), (
            "第二轮必须用没试过的工具，重复查同样的东西没有意义"
        )
        assert TOOL_GRAPH_GLOBAL in second_round_tools

    def test_has_evidence_does_not_loop(self, graph_store):
        """拿到证据就不该再查 —— 多查一轮是纯粹的延迟与成本。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        result = agent.answer("张建国服用了什么药？")

        assert result["tool_rounds"] == 1
        assert _assess_entries(result)[0]["reason"] == "evidence_sufficient_heuristic"

    def test_max_tool_rounds_one_disables_loop(self, graph_store):
        """上限设为 1 时应退化成原来的单轮行为。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=1)
        result = agent.answer("完全不存在的实体名称")

        assert result["tool_rounds"] == 1
        assert _assess_entries(result)[0]["reason"] == "reached_max_rounds"

    def test_loop_stops_when_no_tools_left(self, graph_store):
        """所有工具都试过且全空时必须停下，不能无限转。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=5)
        result = agent.answer("完全不存在的实体名称")

        reasons = [e["reason"] for e in _assess_entries(result)]
        assert reasons[-1] in ("empty_evidence_no_tools_left", "evidence_sufficient_heuristic")
        assert result["tool_rounds"] <= len(ALL_TOOLS)

    def test_kb_meta_never_loops(self, graph_store):
        """元信息问题一轮到底：答案来自知识库统计，补检索毫无意义。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        result = agent.answer("知识库里有几个文档？")

        assert result["tool_rounds"] == 1
        assert _assess_entries(result)[0]["reason"] == "kb_meta_terminal"


# =============================================================================
# 2. 证据累积（多轮改造最容易踩的坑）
# =============================================================================


class TestEvidenceAccumulation:
    def test_second_round_does_not_overwrite_first(self, graph_store):
        """第二轮的证据必须追加，不能覆盖第一轮。

        如果每轮覆盖，第一轮辛苦召回的关系链会被第二轮的社区摘要冲掉，
        多轮反而比单轮更差 —— 这是本改造里最隐蔽的失败模式。
        """
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        result = agent.answer("完全不存在的实体名称")

        rounds = {e["round"] for e in _tool_entries(result)}
        assert rounds == {0, 1}, "两轮的工具调用都应被记录，且轮次编号不同"

    def test_evidence_text_contains_all_rounds(self, graph_store):
        """最终喂给 LLM 的证据文本必须包含两轮的内容。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)

        # 直接驱动节点，观察 evidence_text 的拼接结果
        state = agent._initial_state("完全不存在的实体名称")
        state.update(agent._plan_node(state))
        state.update(agent._assemble_tool_output(state, agent._execute_tools(state)))
        first_round_text = state["evidence_text"]
        first_round_blocks = len(state["evidence_blocks"])

        state.update(agent._assess_node(state))
        state.update(agent._plan_node(state))
        state.update(agent._assemble_tool_output(state, agent._execute_tools(state)))

        assert len(state["evidence_blocks"]) > first_round_blocks, "第二轮必须追加证据块"
        assert first_round_text in state["evidence_text"], "第一轮的证据不能被丢弃"

    def test_sources_deduplicated_across_rounds(self, graph_store):
        """多轮可能重复召回同一个块，来源列表要去重，否则前端会显示重复引用。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        # 强制两轮向量检索
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([
            '{"sufficient": false, "missing": "缺剂量", "tools": ["vector_search"], "query": "剂量"}',
            "答案。",
        ])
        result = agent.answer("张建国的用药剂量")

        previews = [s.get("content_preview") for s in result["sources"]]
        assert len(previews) == len(set(previews)), "同一内容不应在来源里出现两次"


# =============================================================================
# 3. 充分性判官
# =============================================================================


class TestLLMAssessor:
    def test_insufficient_verdict_triggers_extra_round(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([
            '{"sufficient": false, "missing": "缺少研发方", "tools": ["graph_search"], "query": "诺华 研发"}',
            "答案。",
        ])
        result = agent.answer("张建国服用的药是哪家公司研发的？")

        assert result["tool_rounds"] == 2
        assert _assess_entries(result)[0]["reason"] == "llm_judged_insufficient"

    def test_assessor_suggested_tools_are_executed(self, graph_store):
        """判官建议的工具必须被下一轮真正执行 —— 否则它的规划等于白做。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([
            '{"sufficient": false, "missing": "缺机构", "tools": ["graph_global_search"], "query": "涉及哪些机构"}',
            "答案。",
        ])
        result = agent.answer("张建国服用的药是哪家公司研发的？")

        second_plan = _plan_entries(result)[1]
        assert second_plan["tools"] == [TOOL_GRAPH_GLOBAL]
        assert second_plan["decided_by"] == "assessor", "应标记为采纳判官建议"

    def test_assessor_refined_query_is_applied(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([
            '{"sufficient": false, "missing": "缺机构", "tools": ["graph_search"], "query": "诺华公司"}',
            "答案。",
        ])
        result = agent.answer("张建国服用的药是哪家公司研发的？")

        assess = _assess_entries(result)[0]
        assert assess["decision"] == "plan"
        assert assess["missing"] == "缺机构"

    def test_sufficient_verdict_stops_loop(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM(['{"sufficient": true, "missing": ""}', "答案。"])
        result = agent.answer("张建国服用了什么药？")

        assert result["tool_rounds"] == 1
        assert _assess_entries(result)[0]["reason"] == "llm_judged_sufficient"

    def test_assessor_failure_is_conservative(self, graph_store):
        """判官挂了必须保守收敛（按"充分"处理），不能把主流程卡死。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([RuntimeError("判官服务不可用"), "答案。"])
        result = agent.answer("张建国服用了什么药？")

        assert result["success"] is True
        assert result["tool_rounds"] == 1

    def test_malformed_assessor_output_is_conservative(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM(["我看不出够不够", "答案。"])
        result = agent.answer("张建国服用了什么药？")
        assert result["tool_rounds"] == 1

    def test_assessor_cannot_request_kb_meta(self, graph_store):
        """判官不该把元信息工具当作"补充检索"手段 —— 那和当前问题无关。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM([
            '{"sufficient": false, "missing": "x", "tools": ["kb_meta"], "query": ""}',
            "答案。",
        ])
        result = agent.answer("张建国服用了什么药？")

        for entry in _plan_entries(result):
            assert TOOL_KB_META not in entry["tools"], "kb_meta 不应被当作续查手段"

    def test_assessor_disabled_skips_llm_call(self, graph_store):
        """关掉判官就不该有任何额外 LLM 调用。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        result = agent.answer("张建国服用了什么药？")
        assert result["usage"]["llm_calls"] == 1, "只应有生成答案那一次调用"


class TestAssessmentGate:
    """廉价前置门控：只在证据"薄"或聚合型问题时才请判官。

    这组测试固化的是**实测调优结论**：
    初版实现里只要"有未用过的工具"就请判官，结果 14 题里 11 题都补检索，
    平均 2.29 轮，但召回率相比单轮版没有提升，单题延迟却从 4.5s 涨到 16.2s。
    判官天然倾向于"再查一点更保险"，所以必须由零成本的门控先挡一道。
    """

    @pytest.fixture
    def thick_evidence_store(self):
        """证据很厚：单条结果就有几千字符。"""
        long_content = "张建国服用了沙库巴曲缬沙坦钠片，该药由诺华研发。" * 40
        return FakeVectorStore([
            {"score": 0.95, "chunk_id": "c1", "doc_id": "d1", "doc_title": "病历.pdf",
             "content": long_content, "chunk_type": "document", "attributes": {}},
        ])

    def test_thick_evidence_skips_assessor(self, graph_store, thick_evidence_store):
        """证据充足 + 非聚合问题 -> 不调判官，直接生成。"""
        agent = _agent(graph_store, thick_evidence_store, max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM("答案。")

        result = agent.answer("张建国服用了什么药？")

        assert result["tool_rounds"] == 1, "证据够厚时不该再补检索"
        assert result["usage"]["llm_calls"] == 1, "判官不该被调用"
        assert _assess_entries(result)[0]["reason"] == "evidence_sufficient_heuristic"

    def test_thin_evidence_triggers_assessor(self, graph_store):
        """证据薄 -> 请判官复核。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM(['{"sufficient": true, "missing": ""}', "答案。"])

        result = agent.answer("张建国服用了什么药？")

        assert result["usage"]["llm_calls"] >= 2, "证据薄时应调用判官"
        assert _assess_entries(result)[0]["reason"] == "llm_judged_sufficient"

    def test_global_question_assessed_even_with_thick_evidence(
        self, graph_store, thick_evidence_store
    ):
        """聚合型问题即使证据很多也要复核：它需要**完整清单**，部分样本仍是错的。"""
        agent = _agent(graph_store, thick_evidence_store, max_tool_rounds=3)
        agent.use_llm_assessor = True
        agent._llm = FakeLLM(['{"sufficient": true, "missing": ""}', "答案。"])

        result = agent.answer("文档中一共涉及哪些机构？")

        assert _assess_entries(result)[0]["reason"] == "llm_judged_sufficient"

    def test_gate_threshold_is_configurable(self, graph_store):
        """阈值语义：`证据字符数 < thin_evidence_chars` 才算"薄"。

        所以阈值调到极大等价于"永远请判官"，调到 0 等价于"（非聚合问题）从不请"。
        """
        # 阈值极大 -> 什么都算薄 -> 一定请判官
        always = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3,
                        thin_evidence_chars=10_000_000)
        always.use_llm_assessor = True
        always._llm = FakeLLM(['{"sufficient": true, "missing": ""}', "答案。"])
        assert always.answer("张建国服用了什么药？")["usage"]["llm_calls"] >= 2

        # 阈值 0 -> 没有证据会被判为"薄" -> 非聚合问题不请判官
        never = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=3,
                       thin_evidence_chars=0)
        never.use_llm_assessor = True
        never._llm = FakeLLM("答案。")
        assert never.answer("张建国服用了什么药？")["usage"]["llm_calls"] == 1

    def test_gate_does_not_block_empty_evidence_switch(self, graph_store):
        """门控只影响"请不请判官"，不能影响"检索全空时换工具"这条启发式。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3,
                       thin_evidence_chars=0)
        agent.use_llm_assessor = False
        result = agent.answer("完全不存在的实体名称")

        assert result["tool_rounds"] == 2
        assert _assess_entries(result)[0]["reason"] == "empty_evidence_switch_tools"


# =============================================================================
# 4. 规划失败降级链
# =============================================================================


class TestPlanningFallbackChain:
    def test_zero_tool_calls_still_runs_a_tool(self, graph_store):
        """任何情况下都必须至少执行一个工具，否则生成阶段没有任何依据。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        result = agent.answer("张建国服用了什么药？")
        assert _tool_entries(result), "必须至少执行一次工具调用"

    def test_fallback_guard_catches_invalid_heuristic(self, graph_store, monkeypatch):
        """兜底链第 1 层：`plan_route` 若产出非法工具名，必须被拦下并换成兜底组合。

        为什么用 monkeypatch：`plan_route` 按构造只返回合法工具名，
        正常情况下这一层永远不会触发（有 `test_all_routes_are_valid_tool_names`
        守着这个不变量）。但"永远不会触发"的分支最容易在重构后静默失效，
        所以这里主动破坏不变量、验证防御真的有效。
        """
        import app.services.agent_qa as agent_module

        monkeypatch.setattr(agent_module, "plan_route", lambda _q: ["delete_all", "sudo"])

        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        result = agent.answer("张建国服用了什么药？")

        first_plan = _plan_entries(result)[0]
        assert first_plan["decided_by"] == "fallback"
        assert first_plan["fallback_reason"] == "no_valid_tools"
        assert first_plan["tools"] == list(DEFAULT_PLAN_TOOLS)
        assert _tool_entries(result), "兜底后仍然必须真的执行工具"

    def test_fallback_avoids_already_used_tools(self, graph_store):
        """兜底时也要优先挑没试过的工具，而不是原地重复。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        agent.use_llm_router = True
        agent._llm = FakeLLM([
            '{"tools": ["vector_search", "graph_search"]}',   # 第一轮
            '{"tools": ["非法工具"]}',                          # 第二轮：全非法 -> 兜底
            "答案。",
        ])
        result = agent.answer("完全不存在的实体名称")

        second_plan = _plan_entries(result)[1]
        assert second_plan["fallback_reason"] == "no_valid_tools"
        # 默认组合（向量 + 图）都用过了，兜底应挑剩下的未用过工具
        assert second_plan["tools"] == [TOOL_GRAPH_GLOBAL]

    def test_router_error_recorded_in_trace(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        agent.use_llm_router = True
        agent._llm = FakeLLM([RuntimeError("路由超时"), "答案。"])
        result = agent.answer("张建国服用了什么药？")

        first_plan = _plan_entries(result)[0]
        assert first_plan["decided_by"] == "heuristic"
        assert "路由超时" in first_plan["router_error"]

    def test_tool_execution_error_recorded(self, graph_store):
        """单个工具抛异常不能让整轮失败。"""
        agent = _agent(graph_store, FakeVectorStore(raise_on_search=True), max_tool_rounds=2)
        result = agent.answer("张建国服用了什么药？")

        assert result["success"] is True
        vector_entries = [e for e in _tool_entries(result) if e["tool"] == TOOL_VECTOR]
        assert vector_entries[0]["error"] is True

    def test_no_vector_store_still_works(self, graph_store):
        agent = _agent(graph_store, None, max_tool_rounds=2)
        result = agent.answer("张建国服用了什么药？")
        assert result["success"] is True
        assert result["answer"]


class TestFollowupPlanParsing:
    def test_explicit_empty_means_stop(self):
        """模型明确说"不用再查了"应被识别，而不是当成解析失败回退启发式。"""
        parsed = GraphRAGAgent._parse_followup_plan('{"tools": []}', [TOOL_VECTOR])
        assert parsed["explicit_empty"] is True
        assert parsed["tools"] == []

    def test_malformed_means_not_explicit(self):
        parsed = GraphRAGAgent._parse_followup_plan("不是 JSON", [TOOL_VECTOR])
        assert parsed["explicit_empty"] is False

    def test_filters_unavailable_tools(self):
        parsed = GraphRAGAgent._parse_followup_plan(
            '{"tools": ["graph_search", "kb_meta"]}', [TOOL_GRAPH]
        )
        assert parsed["tools"] == [TOOL_GRAPH]

    def test_extracts_refined_query(self):
        parsed = GraphRAGAgent._parse_followup_plan(
            '{"tools": ["graph_search"], "query": "诺华 研发"}', [TOOL_GRAPH]
        )
        assert parsed["query"] == "诺华 研发"


# =============================================================================
# 5. 用量追踪在 Agent 里的接线
# =============================================================================


class TestAgentUsageTracking:
    def test_usage_reported(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        result = agent.answer("张建国服用了什么药？")

        usage = result["usage"]
        assert usage["llm_calls"] >= 1
        assert usage["total_tokens"] > 0
        assert "generate" in usage["by_node"]

    def test_usage_isolated_between_requests(self, graph_store):
        """同一个 Agent 连续两次提问，用量不能累加。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))

        first = agent.answer("张建国服用了什么药？")["usage"]
        second = agent.answer("张建国患有什么病？")["usage"]

        assert first["llm_calls"] == second["llm_calls"], (
            "第二次的用量应与第一次相同；若翻倍说明追踪器被复用，并发场景会算错"
        )

    def test_multi_round_costs_more_calls(self, graph_store):
        """多轮补检索会带来额外调用 —— 这是这个特性的真实代价，必须可见。"""
        single = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), max_tool_rounds=1)
        single_result = single.answer("张建国服用了什么药？")

        multi = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        multi_result = multi.answer("完全不存在的实体名称")

        assert multi_result["tool_rounds"] > single_result["tool_rounds"]

    def test_cost_none_without_price_config(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        assert agent.answer("张建国服用了什么药？")["usage"]["cost"] is None

    def test_cost_computed_with_price_config(self, graph_store):
        agent = _agent(
            graph_store, FakeVectorStore(VECTOR_RESULTS),
            price_input_per_1m=2.0, price_output_per_1m=8.0,
        )
        usage = agent.answer("张建国服用了什么药？")["usage"]
        assert usage["cost_configured"] is True
        assert usage["cost"] is not None and usage["cost"] >= 0

    def test_llm_failure_recorded_with_error(self, graph_store):
        """生成失败也要记账，否则失败请求的成本被静默吞掉。"""
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS), "答案。")
        agent._llm = FakeLLM(RuntimeError("生成失败"))
        result = agent.answer("张建国服用了什么药？")

        assert result["usage"]["failed_calls"] >= 1


class TestStreamUsage:
    def test_stream_emits_usage_event(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        events = list(agent.answer_stream("张建国服用了什么药？"))
        names = [e["event"] for e in events]

        assert "usage" in names
        usage_event = next(e for e in events if e["event"] == "usage")
        assert usage_event["data"]["llm_calls"] >= 1

    def test_stream_emits_assess_event(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        events = list(agent.answer_stream("张建国服用了什么药？"))
        assert any(e["event"] == "assess" for e in events)

    def test_stream_multi_round_emits_multiple_sources(self, graph_store):
        """多轮时 sources 事件应出现多次，前端可以逐轮展示检索进展。"""
        agent = _agent(graph_store, FakeVectorStore(results=[]), max_tool_rounds=3)
        events = list(agent.answer_stream("完全不存在的实体名称"))
        source_events = [e for e in events if e["event"] == "sources"]
        assert len(source_events) == 2
        assert source_events[0]["data"]["round"] == 0
        assert source_events[1]["data"]["round"] == 1

    def test_stream_done_includes_usage(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        events = list(agent.answer_stream("张建国服用了什么药？"))
        done = events[-1]
        assert done["event"] == "done"
        assert done["data"]["usage"]["llm_calls"] >= 1

    def test_stream_generation_failure_recorded(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(VECTOR_RESULTS))
        agent._llm = FakeLLM(RuntimeError("流式生成失败"))
        events = list(agent.answer_stream("张建国服用了什么药？"))
        assert events[-1]["event"] == "done"
        usage_event = next(e for e in events if e["event"] == "usage")
        assert usage_event["data"]["failed_calls"] >= 1


# =============================================================================
# 6. 辅助函数
# =============================================================================


class TestDigestEvidence:
    def test_empty_blocks(self):
        assert "暂无" in _digest_evidence([])

    def test_marks_empty_blocks(self):
        digest = _digest_evidence([{"tool": TOOL_VECTOR, "round": 0, "empty": True, "text": ""}])
        assert "未召回内容" in digest

    def test_marks_error_blocks(self):
        digest = _digest_evidence(
            [{"tool": TOOL_VECTOR, "round": 0, "empty": True, "error": "连接失败", "text": ""}]
        )
        assert "执行失败" in digest

    def test_truncates_long_text(self):
        digest = _digest_evidence(
            [{"tool": TOOL_GRAPH, "round": 0, "empty": False, "text": "内容" * 500}],
            max_chars_per_block=50,
        )
        assert len(digest) < 400
        assert "…" in digest

    def test_includes_round_number(self):
        digest = _digest_evidence(
            [{"tool": TOOL_GRAPH, "round": 1, "empty": False, "text": "证据"}]
        )
        assert "第2轮" in digest


class TestParseJsonObject:
    @pytest.mark.parametrize("raw,expected", [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('好的：\n{"a": 1}\n希望有帮助', {"a": 1}),
    ])
    def test_valid_inputs(self, raw, expected):
        assert _parse_json_object(raw) == expected

    @pytest.mark.parametrize("raw", ["", "不是 JSON", "[1,2,3]", "{坏}", None])
    def test_invalid_inputs(self, raw):
        assert _parse_json_object(raw) == {}
