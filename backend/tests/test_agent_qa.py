"""Agentic 问答服务的单元测试

测试目标：把 LangGraph 的整条闭环（规划 → 工具 → 生成 → 校验 → 改写重试）
在**没有 API Key、不花钱、确定性**的条件下完整验证。

手法：
- `FakeLLM`：脚本化响应，可按调用序返回不同内容（用于模拟"第一次不可接地、第二次改好"）
- `FakeVectorStore`：实现 `search()` 与 `get_kb_summary()` 两个 Agent 真正调用的方法

这里同时验证一个重要的工程属性：**Agent 在没有任何外部依赖时也能优雅降级**
（向量库为 None、LLM 路由失败、检索报错），而不是抛异常。
"""

import json

import pytest

from app.services.agent_qa import (
    ALL_TOOLS,
    TOOL_GRAPH,
    TOOL_GRAPH_GLOBAL,
    TOOL_KB_META,
    TOOL_VECTOR,
    GraphRAGAgent,
    make_tool_calls,
    plan_route,
    reset_graph_agent,
    verify_citations,
)
from app.services.graph_store import KnowledgeGraphStore, RelationTriple


# =============================================================================
# 测试替身
# =============================================================================


class FakeLLM:
    """脚本化假 LLM。

    - `responses`: 字符串 / 字符串列表 / 异常对象；按调用顺序消费，用尽后重复最后一项
    - 同时提供 `invoke` 与 `stream`，因为 Agent 两条路径都会用到
    """

    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else [responses]
        self.calls = []

    def _next(self):
        if not self.responses:
            return ""
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    def invoke(self, messages):
        self.calls.append(messages)
        item = self._next()
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    def stream(self, messages):
        self.calls.append(messages)
        item = self._next()
        if isinstance(item, Exception):
            raise item
        for piece in _chunks(item):
            yield _Resp(piece)

    @property
    def call_count(self):
        return len(self.calls)


class _Resp:
    def __init__(self, content):
        self.content = content


def _chunks(text, size=8):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


class FakeVectorStore:
    """只实现 Agent 真正调用的两个方法，足以驱动整条链路。"""

    def __init__(self, results=None, summary=None, raise_on_search=False):
        self.results = results if results is not None else []
        self.summary = summary if summary is not None else {
            "document_count": 2, "chunk_count": 17,
            "documents": [{"doc_id": "d1", "title": "病历.pdf"}, {"doc_id": "d2", "title": "指南.pdf"}],
        }
        self.raise_on_search = raise_on_search
        self.search_calls = []

    def search(self, query, top_k=5, **kwargs):
        self.search_calls.append((query, top_k))
        if self.raise_on_search:
            raise RuntimeError("向量库连接失败")
        return self.results[:top_k]

    def get_kb_summary(self):
        return self.summary


# =============================================================================
# 测试用图谱与向量结果
# =============================================================================

GRAPH_TEXT = (
    "患者张建国，男，62岁。诊断为扩张型心肌病。"
    "左室射血分数（LVEF）38%。给予沙库巴曲缬沙坦钠片治疗。"
)


@pytest.fixture
def graph_store():
    store = KnowledgeGraphStore()
    store.add_extractions(
        "d1", "病历.pdf",
        [
            {"extraction_class": "患者", "extraction_text": "张建国",
             "attributes": {"性别": "男"}, "char_interval": {"start_pos": 2, "end_pos": 5}},
            {"extraction_class": "诊断", "extraction_text": "扩张型心肌病",
             "attributes": {}, "char_interval": {"start_pos": 10, "end_pos": 16}},
            {"extraction_class": "药物", "extraction_text": "沙库巴曲缬沙坦钠片",
             "attributes": {}, "char_interval": {"start_pos": 45, "end_pos": 55}},
            {"extraction_class": "机构", "extraction_text": "诺华",
             "attributes": {}, "char_interval": {"start_pos": 60, "end_pos": 62}},
        ],
    )
    store.add_triples(
        [
            RelationTriple("张建国", "患有", "扩张型心肌病", evidence="诊断为扩张型心肌病", confidence=0.9),
            RelationTriple("扩张型心肌病", "治疗", "沙库巴曲缬沙坦钠片",
                           evidence="给予沙库巴曲缬沙坦钠片治疗", confidence=0.88),
            RelationTriple("沙库巴曲缬沙坦钠片", "研发", "诺华",
                           evidence="诺华研发的沙库巴曲缬沙坦钠", confidence=0.85),
        ],
        doc_id="d1", doc_title="病历.pdf", auto_create_entities=False,
    )
    return store


@pytest.fixture
def vector_results():
    return [
        {"score": 0.91, "chunk_id": "c1", "doc_id": "d1", "doc_title": "病历.pdf",
         "content": "患者张建国，男，62岁。诊断为扩张型心肌病。LVEF 38%。",
         "chunk_type": "document", "attributes": {"paragraph_index": 0}},
        {"score": 0.85, "chunk_id": "c2", "doc_id": "d1", "doc_title": "病历.pdf",
         "content": "给予沙库巴曲缬沙坦钠片 50mg bid 治疗。诺华研发。",
         "chunk_type": "document", "attributes": {"paragraph_index": 1}},
    ]


# =============================================================================
# 启发式路由
# =============================================================================


class TestPlanRoute:
    def test_meta_question_uses_meta_tool_only(self):
        assert plan_route("知识库里有几个文档？") == [TOOL_KB_META]

    def test_global_question_uses_community_search(self):
        route = plan_route("文档里共涉及哪些机构？")
        assert TOOL_GRAPH_GLOBAL in route
        assert TOOL_VECTOR in route

    def test_detail_question_uses_vector_and_graph(self):
        """细节类问题默认双路互补：向量管细节，图管跨段关系。"""
        route = plan_route("张建国的射血分数是多少？")
        assert TOOL_VECTOR in route and TOOL_GRAPH in route

    def test_empty_question_falls_back_to_vector(self):
        assert plan_route("") == [TOOL_VECTOR]

    def test_all_routes_are_valid_tool_names(self):
        for query in ["几个文档", "涉及哪些机构", "血压是多少", ""]:
            for name in plan_route(query):
                assert name in ALL_TOOLS, f"路由产生了非法工具名 {name}"


class TestMakeToolCalls:
    def test_shape_matches_langchain_contract(self):
        calls = make_tool_calls("问题", [TOOL_VECTOR, TOOL_GRAPH], hops=3, top_k=7)
        assert len(calls) == 2
        for call in calls:
            assert set(call) >= {"name", "args", "id", "type"}
            assert call["type"] == "tool_call"
            assert call["id"], "id 不能为空，ToolNode 依赖它把结果与调用配对"
        assert calls[0]["args"]["top_k"] == 7
        assert calls[1]["args"]["hops"] == 3

    def test_unknown_tool_skipped(self):
        calls = make_tool_calls("问题", ["不存在工具"])
        assert calls == []

    def test_kb_meta_takes_no_args(self):
        calls = make_tool_calls("问题", [TOOL_KB_META])
        assert calls[0]["args"] == {}


# =============================================================================
# 引用校验 —— Agent 的反思依据
# =============================================================================


class TestVerifyCitations:
    def test_all_grounded_passes(self):
        result = verify_citations(
            answer="患者LVEF为38%。",
            evidence_text="检查显示 LVEF 38%，明显下降。",
        )
        assert result["ungrounded_numbers"] == []
        assert result["passed"] is True

    def test_hallucinated_number_detected(self):
        """答案里的数字在证据里找不到 -> 必须被抓出来。

        这是 RAG 最危险的失败模式：检索到的证据里是 38%，模型却写成 48%。
        """
        result = verify_citations(
            answer="患者LVEF为48%。",
            evidence_text="检查显示 LVEF 38%，明显下降。",
        )
        assert "48%" in result["ungrounded_numbers"]
        assert result["passed"] is False

    def test_whitespace_difference_tolerated(self):
        """证据里写成"38 %"、答案里写"38%"，不应误报。"""
        result = verify_citations(answer="LVEF 38%", evidence_text="LVEF 38 %")
        assert result["ungrounded_numbers"] == []

    def test_single_digit_numbers_not_checked(self):
        """个位数太常见，校验没有意义，必须排除以免噪声拉低指标。"""
        result = verify_citations(answer="共 5 项指标", evidence_text="完全没有数字的证据")
        assert result["checked_numbers"] == []

    def test_numbers_skipped_when_evidence_has_none(self):
        """证据里本来就没有数字时，不应把所有数字判为不可接地。"""
        result = verify_citations(
            answer="患者有 3 个症状，病程 12 个月。",
            evidence_text="患者主诉胸闷、气短、水肿。",
        )
        assert result["checked_numbers"] == []
        assert result["grounded_ratio"] == 1.0

    def test_ungrounded_entity_detected(self):
        result = verify_citations(
            answer="患者服用了【阿司匹林】。",
            evidence_text="患者服用了沙库巴曲缬沙坦钠片。",
            known_entities=["阿司匹林", "沙库巴曲缬沙坦钠片"],
        )
        assert "阿司匹林" in result["ungrounded_entities"]

    def test_entity_not_mentioned_in_answer_not_checked(self):
        """答案没提到的实体不参与校验，否则分母会被无关实体灌水。"""
        result = verify_citations(
            answer="患者服用了沙库巴曲缬沙坦钠片。",
            evidence_text="患者服用了沙库巴曲缬沙坦钠片。",
            known_entities=["沙库巴曲缬沙坦钠片", "从未提到的实体"],
        )
        assert result["checked_entities"] == ["沙库巴曲缬沙坦钠片"]

    def test_empty_answer_is_vacuously_grounded(self):
        result = verify_citations(answer="", evidence_text="任意证据")
        assert result["checked_total"] == 0
        assert result["grounded_ratio"] == 1.0

    def test_ratio_computed_over_all_checks(self):
        result = verify_citations(
            answer="张建国 48% 服用了 阿司匹林。",
            evidence_text="张建国 38% 服用了沙库巴曲缬沙坦钠片。",
            known_entities=["张建国", "阿司匹林"],
        )
        # 数字 48% 不可接地，实体 张建国 可接地，阿司匹林 不可接地 -> 1/3
        assert result["grounded_ratio"] == pytest.approx(1 / 3, abs=0.01)
        assert result["passed"] is False


# =============================================================================
# Agent 端到端
# =============================================================================


def _agent(graph_store, vector_store, responses, **kwargs):
    """构造确定性 Agent。

    默认关掉两个 LLM 决策点（路由复核 + 充分性判官）：
    这两个节点各自会多消费一次 LLM 调用，在脚本化 FakeLLM 下会打乱响应顺序，
    让"测反思循环"的用例意外测到"测判官"。需要测判官的用例自己显式打开。
    """
    kwargs.setdefault("use_llm_router", False)
    kwargs.setdefault("use_llm_assessor", False)
    return GraphRAGAgent(
        graph_store=graph_store,
        vector_store=vector_store,
        llm=FakeLLM(responses),
        **kwargs,
    )


class TestAgentEndToEnd:
    def test_answer_uses_vector_and_graph(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "张建国患有扩张型心肌病，LVEF 38%。")
        result = agent.answer("张建国的诊断是什么？")

        assert result["success"] is True
        assert result["attempts"] == 1
        assert TOOL_VECTOR in result["route"] and TOOL_GRAPH in result["route"]
        assert len(result["sources"]) == 2, "向量来源必须回传给前端"
        assert result["graph"], "图检索结果必须回传"

    def test_sources_carry_char_interval_for_highlighting(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        result = agent.answer("张建国")
        # 向量来源的属性里带 paragraph_index，图来源的实体带字符区间
        assert any("chunk_type" in s for s in result["sources"])

    def test_meta_question_routes_to_meta_tool(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "知识库里有 2 个文档。")
        result = agent.answer("知识库里有几个文档？")

        assert result["route"] == [TOOL_KB_META]
        tools_used = [t["tool"] for t in result["trace"] if t.get("node") == "tools"]
        assert TOOL_KB_META in tools_used

    def test_global_question_uses_community_search(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "共涉及 3 类实体。")
        result = agent.answer("文档里共涉及哪些机构？")
        assert TOOL_GRAPH_GLOBAL in result["route"]

    def test_trace_records_every_node(self, graph_store, vector_results):
        """可观测性：trace 必须能复现"这次问了什么、用了什么工具、校验结果如何"。"""
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        result = agent.answer("张建国")

        nodes = [entry["node"] for entry in result["trace"]]
        assert "plan" in nodes and "tools" in nodes and "generate" in nodes and "verify" in nodes
        plan_entry = next(e for e in result["trace"] if e["node"] == "plan")
        assert plan_entry["decided_by"] == "heuristic"
        assert plan_entry["tools"]

    def test_verification_in_result(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案里没有数字。")
        result = agent.answer("张建国")
        assert "grounded_ratio" in result["verification"]


class TestReflectionLoop:
    """反思循环：校验不通过 -> 改写查询 -> 重试。这是"Agent"与"直线 RAG"的分界。"""

    def test_retry_triggered_by_ungrounded_answer(self, graph_store, vector_results):
        # 第一次答案含幻觉数字 48%，第二次答案接地
        llm = FakeLLM(["患者LVEF为48%。", "患者LVEF为38%。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=False, use_llm_assessor=False, max_attempts=2)
        result = agent.answer("张建国的射血分数是多少？")

        nodes = [entry["node"] for entry in result["trace"]]
        assert "rewrite" in nodes, "校验不通过必须触发改写重试"
        assert result["attempts"] == 2
        assert result["verification"]["passed"] is True, "重试后应通过校验"

    def test_no_retry_when_grounded(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "患者LVEF为38%。")
        result = agent.answer("张建国的射血分数是多少？")

        nodes = [entry["node"] for entry in result["trace"]]
        assert "rewrite" not in nodes
        assert result["attempts"] == 1

    def test_retry_stops_at_max_attempts(self, graph_store, vector_results):
        """模型一直幻觉时必须停下，不能无限重试烧钱。"""
        agent = GraphRAGAgent(
            graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
            llm=FakeLLM("患者LVEF为48%。"), use_llm_router=False, use_llm_assessor=False, max_attempts=3,
        )
        result = agent.answer("张建国的射血分数是多少？")

        assert result["attempts"] == 3
        assert result["verification"]["passed"] is False
        assert "48%" in result["verification"]["ungrounded_numbers"]

    def test_rewrite_includes_missing_keywords(self, graph_store, vector_results):
        """确定性改写要把"上次没被证据支撑的内容"补进查询。

        这是无 LLM 时的兜底路径，也是评测脚本能跑出稳定结果的前提。
        """
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")

        rewritten = agent._deterministic_rewrite("射血分数是多少", ["48%"])
        assert "48%" in rewritten, "未接地的关键词必须进入新查询"
        assert rewritten.startswith("射血分数是多少")

    def test_deterministic_rewrite_strips_question_words_when_no_keywords(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        rewritten = agent._deterministic_rewrite("请问这个病是什么呢？", [])
        assert "请问" not in rewritten
        assert "呢" not in rewritten
        assert rewritten, "改写结果不能为空，否则重试会检索到无关内容"

    def test_deterministic_rewrite_never_returns_empty(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        # 全是疑问词的极端输入也要能返回非空查询
        assert agent._deterministic_rewrite("？？？", []) != ""

    def test_rewrite_llm_failure_uses_deterministic_fallback(self, graph_store, vector_results):
        """改写用的 LLM 挂了也要能继续，不能整个 Agent 崩掉。"""
        llm = FakeLLM(["患者LVEF为48%。", RuntimeError("改写服务不可用"), "患者LVEF为38%。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=False, use_llm_assessor=False, max_attempts=2)
        result = agent.answer("射血分数是多少")
        assert result["attempts"] == 2
        assert "rewrite" in [e["node"] for e in result["trace"]]


class TestGracefulDegradation:
    """任何一个外部依赖挂掉，Agent 都不应整体失败。"""

    def test_no_vector_store(self, graph_store):
        agent = _agent(graph_store, None, "据现有资料无法回答。")
        result = agent.answer("张建国")
        assert result["success"] is True
        tools_used = [t["tool"] for t in result["trace"] if t.get("node") == "tools"]
        assert TOOL_VECTOR in tools_used, "向量工具仍应被调用（返回错误而不是抛异常）"

    def test_vector_store_raises(self, graph_store):
        agent = _agent(graph_store, FakeVectorStore(raise_on_search=True), "据现有资料无法回答。")
        result = agent.answer("张建国")
        assert result["success"] is True

    def test_empty_graph(self, vector_results):
        agent = _agent(KnowledgeGraphStore(), FakeVectorStore(vector_results), "答案。")
        result = agent.answer("张建国")
        assert result["success"] is True

    def test_generation_error_reported_not_raised(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), RuntimeError("生成失败"))
        result = agent.answer("张建国")
        assert result["success"] is True
        assert "出错" in result["answer"]

    def test_llm_router_failure_falls_back_to_heuristic(self, graph_store, vector_results):
        agent = GraphRAGAgent(
            graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
            llm=FakeLLM([RuntimeError("路由失败"), "答案。"]),
            use_llm_router=True,
            use_llm_assessor=False,
        )
        result = agent.answer("张建国的诊断是什么？")
        plan_entry = next(e for e in result["trace"] if e["node"] == "plan")
        assert plan_entry["decided_by"] == "heuristic", "路由失败必须回退，且标记出来"
        assert TOOL_VECTOR in result["route"]


class TestLLMRouter:
    def test_llm_route_accepted_when_valid(self, graph_store, vector_results):
        llm = FakeLLM(['{"tools": ["graph_search"]}', "答案。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=True, use_llm_assessor=False)
        result = agent.answer("张建国和诺华什么关系？")

        plan_entry = next(e for e in result["trace"] if e["node"] == "plan")
        assert plan_entry["decided_by"] == "llm"
        assert result["route"] == [TOOL_GRAPH]

    def test_invalid_tool_names_filtered(self, graph_store, vector_results):
        """模型编造工具名时必须被白名单挡掉，否则 ToolNode 会报错。

        断言看的是**首次规划**的结果，而不是累计 route：
        "问题"这个查询在图谱里命中不到实体，图检索会返回空，
        多轮逻辑随后正确地换了未用过的工具再查一轮 —— 那属于多轮行为，
        由 `TestMultiRoundTools` 单独覆盖，不该混进工具名白名单的用例里。
        """
        llm = FakeLLM(['{"tools": ["graph_search", "delete_everything"]}', "答案。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=True, use_llm_assessor=False)
        result = agent.answer("问题")

        first_plan = next(e for e in result["trace"] if e["node"] == "plan")
        assert first_plan["tools"] == [TOOL_GRAPH], "非法工具名必须被过滤掉"
        assert "delete_everything" not in result["route"]

    def test_all_invalid_falls_back_to_heuristic(self, graph_store, vector_results):
        llm = FakeLLM(['{"tools": ["不存在的工具"]}', "答案。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=True, use_llm_assessor=False)
        result = agent.answer("张建国")
        plan_entry = next(e for e in result["trace"] if e["node"] == "plan")
        assert plan_entry["decided_by"] == "heuristic"
        assert TOOL_VECTOR in result["route"]

    @pytest.mark.parametrize("raw", [
        "", "不是 JSON", '{"tools": "graph_search"}', '{"tools": null}', "[]", "{坏 json}",
    ])
    def test_malformed_router_output_returns_empty(self, raw):
        assert GraphRAGAgent._parse_tool_choice(raw) == []

    def test_fenced_json_router_output(self):
        raw = '```json\n{"tools": ["vector_search"]}\n```'
        assert GraphRAGAgent._parse_tool_choice(raw) == [TOOL_VECTOR]

    def test_meta_question_bypasses_llm_router(self, graph_store, vector_results):
        """元信息问题不该浪费一次 LLM 路由调用。"""
        llm = FakeLLM(["答案。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=True, use_llm_assessor=False)
        result = agent.answer("知识库有几个文档？")
        assert result["route"] == [TOOL_KB_META]
        assert llm.call_count == 1, "只应有生成答案那一次调用"


# =============================================================================
# 流式
# =============================================================================


class TestStreaming:
    def _events(self, agent, question):
        return list(agent.answer_stream(question))

    def test_event_sequence(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "张建国LVEF为38%。")
        events = [e["event"] for e in self._events(agent, "张建国")]

        assert events[0] == "sources", "先给来源，用户能立刻看到检索到了什么"
        assert "chunk" in events
        assert events[-1] == "done"

    def test_sources_sent_before_answer(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        events = self._events(agent, "张建国")

        sources_event = next(e for e in events if e["event"] == "sources")
        assert len(sources_event["data"]["sources"]) == 2
        assert sources_event["data"]["route"]

    def test_graph_event_emitted(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        events = self._events(agent, "张建国")
        graph_events = [e for e in events if e["event"] == "graph"]
        assert graph_events, "图检索结果应单独作为事件推给前端"
        assert graph_events[0]["data"]["mode"] == "local"

    def test_chunks_reassemble_into_answer(self, graph_store, vector_results):
        expected = "张建国LVEF为38%，诊断为扩张型心肌病。"
        agent = _agent(graph_store, FakeVectorStore(vector_results), expected)
        chunks = [e["data"]["content"] for e in self._events(agent, "张建国") if e["event"] == "chunk"]
        assert "".join(chunks) == expected

    def test_verification_event_emitted(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "张建国LVEF为38%。")
        events = self._events(agent, "张建国")
        assert any(e["event"] == "verification" for e in events)

    def test_retry_event_on_ungrounded(self, graph_store, vector_results):
        llm = FakeLLM(["患者LVEF为48%。", "患者LVEF为38%。"])
        agent = GraphRAGAgent(graph_store=graph_store, vector_store=FakeVectorStore(vector_results),
                              llm=llm, use_llm_router=False, use_llm_assessor=False, max_attempts=2)
        events = [e["event"] for e in agent.answer_stream("射血分数是多少")]
        assert "retry" in events

    def test_done_event_reports_attempts(self, graph_store, vector_results):
        agent = _agent(graph_store, FakeVectorStore(vector_results), "答案。")
        events = self._events(agent, "张建国")
        done = events[-1]
        assert done["data"]["success"] is True
        assert done["data"]["attempts"] == 1

    def test_stream_with_no_vector_store(self, graph_store):
        agent = _agent(graph_store, None, "无法回答。")
        events = [e["event"] for e in agent.answer_stream("张建国")]
        assert events[-1] == "done"


# =============================================================================
# 单例
# =============================================================================


class TestSingleton:
    def test_same_store_reuses_agent(self, graph_store, vector_results):
        reset_graph_agent()
        from app.services.agent_qa import get_graph_agent

        first = get_graph_agent(graph_store, FakeVectorStore(vector_results))
        second = get_graph_agent(graph_store, FakeVectorStore(vector_results))
        assert first is second

    def test_new_store_rebuilds_agent(self, graph_store, vector_results):
        """换图谱必须重建：Agent 内部的 GraphRetriever 捕获了旧 store。"""
        reset_graph_agent()
        from app.services.agent_qa import get_graph_agent

        first = get_graph_agent(graph_store, FakeVectorStore(vector_results))
        second = get_graph_agent(KnowledgeGraphStore(), FakeVectorStore(vector_results))
        assert first is not second
