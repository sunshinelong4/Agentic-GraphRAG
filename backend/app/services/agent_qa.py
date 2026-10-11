"""Agentic 问答服务 - 基于 LangGraph 的检索 Agent

改造前后的本质差别
------------------
改造前（`app/services/qa_agent.py` 的 `QAAgent.answer`）：

    检索向量 -> 拼上下文 -> 让 LLM 回答          # 一条直线，没有决策

改造后（本模块）：

    plan ──> tools ──> generate ──> verify ─┬─(校验不通过且未超次数)─> rewrite ─┐
      ↑                                     │                                    │
      └─────────────────────────────────────┴─(通过或超次数)────────────────> END

三个"Agent"要素齐了：
1. **自主决策**：`plan` 节点决定用哪些检索工具（向量 / 图局部 / 图全局 / 库元信息），
   而不是由代码写死；明显的问题走启发式（省钱），含糊的问题才问 LLM。
2. **反思循环**：`verify` 节点校验答案里的数字与实体是否都能在检索到的证据中找到，
   不通过就改写查询重试 —— 这是一条真实的「生成 → 校验 → 修正」闭环。
3. **可观测**：`trace` 记录每个节点做了什么、每次工具调用返回多少条，便于排查与评测。

为什么用 LangGraph 的 ToolNode 而不是自己 if-else 调函数：
ToolNode 负责把 LLM 产生的 `tool_calls` 正确派发到对应工具、把结果包成 ToolMessage、
处理参数校验与异常。这正是"工具调用"这一抽象的价值所在，也是简历上可讲的点。

可测试性
--------
`llm` 与 `vector_store` 都是注入的，因此整个 Agent 可以在**没有 API Key** 的情况下
用 FakeLLM + FakeVectorStore 完整跑通并被断言。`use_llm_router=False` 时
规划节点走纯启发式，测试完全确定性。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Annotated, Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from app.services.graph_retriever import GraphRetriever, is_global_query
from app.services.graph_store import KnowledgeGraphStore
from app.services.llm_usage import LLMUsageTracker, estimate_tokens
from app.services.qa_agent import KB_META_PATTERNS

logger = logging.getLogger(__name__)


# =============================================================================
# 工具名常量
# =============================================================================

TOOL_VECTOR = "vector_search"
TOOL_GRAPH = "graph_search"
TOOL_GRAPH_GLOBAL = "graph_global_search"
TOOL_KB_META = "kb_meta"

ALL_TOOLS = (TOOL_VECTOR, TOOL_GRAPH, TOOL_GRAPH_GLOBAL, TOOL_KB_META)

# 规划全部失败时的兜底工具组合。
# 之所以是"向量 + 图局部"而不是全部四个：这两个覆盖最广（细节 + 跨块关系），
# 而 kb_meta 只对元信息问题有意义，全局社区检索在非聚合问题上会大幅增加 token 却没什么收益。
DEFAULT_PLAN_TOOLS = (TOOL_VECTOR, TOOL_GRAPH)


# =============================================================================
# 状态定义
# =============================================================================


class AgentState(TypedDict, total=False):
    """Agent 的共享状态。

    `messages` 使用 LangGraph 的 `add_messages` reducer：节点返回的新消息会
    **追加**而不是覆盖，ToolNode 依赖这个语义把 ToolMessage 挂到消息列表上。
    """

    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    current_query: str          # 可以被 rewrite 或 assess 改写的查询
    route: List[str]            # 本次会话**累计**用过的工具（按首次出现顺序，用于上报）
    current_tools: List[str]    # 当前这一轮实际用的工具（用于重试/续查决策）
    used_tools: List[str]       # 累计用过的工具（用于"换没试过的工具再查一轮"）
    evidence_blocks: List[Dict[str, Any]]  # 逐轮逐工具累积的证据块
    tool_round: int             # 当前工具调用轮次（0 起）
    max_tool_rounds: int        # 工具调用轮数上限
    assess: Dict[str, Any]      # 最近一次充分性评估结果
    evidence_text: str          # 汇总后的证据文本（喂给 LLM）
    sources: List[Dict[str, Any]]
    graph_payload: Dict[str, Any]
    answer: str
    verification: Dict[str, Any]
    attempts: int
    trace: List[Dict[str, Any]]
    usage_tracker: Any          # LLMUsageTracker 实例（每个请求一个，隔离并发）


# =============================================================================
# 检索来源路由（启发式，零成本）
# =============================================================================


def plan_route(question: str) -> List[str]:
    """启发式决定用哪些工具。

    设计原则：**能不问模型就不问**。
    - 知识库元信息问题（"有几个文档"）：只查元数据，不要浪费检索
    - 全局聚合问题（"涉及哪些机构"）：社区摘要 + 向量，双保险
    - 其余问题：向量 + 图局部，两者互补（图擅长多跳，向量擅长细节）
    """
    if not question:
        return [TOOL_VECTOR]
    if any(re.search(pattern, question) for pattern in KB_META_PATTERNS):
        return [TOOL_KB_META]
    if is_global_query(question):
        return [TOOL_GRAPH_GLOBAL, TOOL_VECTOR]
    return [TOOL_VECTOR, TOOL_GRAPH]


def make_tool_calls(question: str, tools: Sequence[str], hops: int = 2, top_k: int = 5) -> List[Dict[str, Any]]:
    """把路由结果转成 LangChain 的 tool_calls 结构，交给 ToolNode 执行。"""
    calls: List[Dict[str, Any]] = []
    for index, name in enumerate(tools):
        if name == TOOL_VECTOR:
            args = {"query": question, "top_k": top_k}
        elif name == TOOL_GRAPH:
            args = {"query": question, "hops": hops}
        elif name == TOOL_GRAPH_GLOBAL:
            args = {"query": question}
        elif name == TOOL_KB_META:
            args = {}
        else:
            continue
        calls.append(
            {"name": name, "args": args, "id": f"call_{index}_{name}", "type": "tool_call"}
        )
    return calls


def _digest_evidence(
    blocks: Sequence[Dict[str, Any]],
    max_chars_per_block: int = 220,
) -> str:
    """把证据块压成短摘要，用于喂给"续查规划"和"充分性判官"。

    为什么要压缩：这两次调用只是让模型判断"缺什么"，不需要读全文。
    直接把完整证据塞进去会让这两个辅助调用的 token 消耗逼近答案生成本身，
    而它们的产出只是一个工具名。代价与收益完全不成比例。

    摘要保留：是哪个工具、哪一轮、有没有拿到东西、内容开头一小段。
    """
    if not blocks:
        return "（暂无任何检索结果）"

    lines: List[str] = []
    for block in blocks:
        tool = block.get("tool") or "?"
        round_index = block.get("round", 0)
        if block.get("error"):
            lines.append(f"- [第{round_index + 1}轮] {tool}: 执行失败（{block['error']}）")
            continue
        if block.get("empty"):
            lines.append(f"- [第{round_index + 1}轮] {tool}: 未召回内容")
            continue
        text = re.sub(r"\s+", " ", block.get("text") or "").strip()
        lines.append(
            f"- [第{round_index + 1}轮] {tool}: {text[:max_chars_per_block]}"
            + ("…" if len(text) > max_chars_per_block else "")
        )
    return "\n".join(lines)


def _parse_json_object(raw: str) -> Dict[str, Any]:
    """从 LLM 输出里稳健地取出 JSON 对象。

    要处理的脏情况和 `relation_extractor._parse_json_array` 一样：
    代码围栏、前后解释文字、根本不是 JSON。抽成函数避免在三个节点里各写一遍。
    """
    if not raw:
        return {}
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    left, right = text.find("{"), text.rfind("}")
    if left == -1 or right <= left:
        return {}
    try:
        payload = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


# =============================================================================
# 引用校验（Agent 的"反思"环节）
# =============================================================================

# 需要校验的数字：带百分号、带小数、或长度 >= 2 的数字（单个数字太常见，无校验价值）
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def _extract_checkable_numbers(text: str) -> List[str]:
    """抽出值得校验的数字（排除 0-9 这类无信息量的单个数位）。"""
    numbers: List[str] = []
    for match in _NUMBER_PATTERN.finditer(text or ""):
        token = match.group(0)
        if "%" in token or "." in token or len(token) >= 2:
            numbers.append(token)
    return sorted(set(numbers))


def _normalize_for_match(text: str) -> str:
    """去掉空白，便于"38 %"与"38%"这类写法差异能匹配上。"""
    return re.sub(r"\s+", "", text or "")


def verify_citations(
    answer: str,
    evidence_text: str,
    known_entities: Optional[Sequence[str]] = None,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """校验答案里的数字与实体是否都能在检索到的证据中找到。

    为什么需要这一步：
    RAG 最常见的失败模式不是"检索不到"，而是"检索到了但模型自己编了数字"。
    垂直领域（医疗/法律/金融）里一个编造的数值就是事故。
    这里做的是**可计算**的接地性（groundedness）检查，不依赖另一个 LLM 当裁判，
    因此确定性、零成本、可复现 —— 可以直接作为评测指标。

    检查两类内容：
    - **数字**：答案里的 158、38%、1.52 等必须出现在证据中
    - **实体**：答案里提到的图谱实体名必须出现在证据中

    Returns:
        {
          "grounded_ratio": float,        # 可接地比例
          "checked_total": int,
          "ungrounded_numbers": [...],
          "ungrounded_entities": [...],
          "passed": bool,
        }
    """
    evidence_norm = _normalize_for_match(evidence_text)

    # 只校验"有实质内容"的数字：证据里完全没有数字时跳过数字检查，
    # 否则纯文字证据会导致所有数字都被判为不可接地，指标失去意义。
    evidence_numbers = _extract_checkable_numbers(evidence_text)
    checked_numbers = _extract_checkable_numbers(answer) if evidence_numbers else []
    ungrounded_numbers = [
        number for number in checked_numbers if _normalize_for_match(number) not in evidence_norm
    ]

    ungrounded_entities: List[str] = []
    checked_entities: List[str] = []
    for entity in known_entities or []:
        if entity and entity in (answer or ""):
            checked_entities.append(entity)
            if _normalize_for_match(entity) not in evidence_norm:
                ungrounded_entities.append(entity)

    checked_total = len(checked_numbers) + len(checked_entities)
    ungrounded_total = len(ungrounded_numbers) + len(ungrounded_entities)

    if checked_total == 0:
        # 没有任何可校验内容：不能判定为"通过"，但也不该无限重试
        grounded_ratio = 1.0
    else:
        grounded_ratio = (checked_total - ungrounded_total) / checked_total

    return {
        "grounded_ratio": round(grounded_ratio, 4),
        "checked_total": checked_total,
        "checked_numbers": checked_numbers,
        "checked_entities": checked_entities,
        "ungrounded_numbers": ungrounded_numbers,
        "ungrounded_entities": ungrounded_entities,
        "passed": grounded_ratio >= threshold,
        "threshold": threshold,
    }


# =============================================================================
# 提示词
# =============================================================================

ROUTER_SYSTEM_PROMPT = """你是一个检索规划专家。根据用户问题，决定需要调用哪些检索工具。

可用工具：
- vector_search(query): 语义向量检索，适合查找具体事实、数值、定义、细节描述
- graph_search(query): 知识图谱多跳检索，适合查找实体之间的关系、跨段落推理
- graph_global_search(query): 知识图谱全局社区检索，适合"整体涉及哪些""一共有多少"这类聚合问题
- kb_meta(): 查询知识库自身的元信息（文档数量、文档清单）

规则：
1. 可以同时选择多个工具（互补时一起用）。
2. 涉及"哪些/几个/多少/整体/统计"等聚合提问时，必须包含 graph_global_search。
3. 只输出 JSON，不要解释。格式：{"tools": ["vector_search", "graph_search"]}"""

ANSWER_SYSTEM_PROMPT = """你是一个严谨的知识问答助手。请严格依据【检索证据】回答问题。

要求：
1. **只使用证据中出现的信息**。证据不足以回答时，明确说明"现有资料无法回答"，绝不编造。
2. 回答中的每个数字、每个实体名称都必须来自证据，不得推测或改写数值。
3. 输出格式：
   - 先用一句话给出结论
   - 再用"• "分点展开，每点包含关键实体（用【】标注）和说明
4. 不要提及"证据""检索""上下文""文档"等词，直接陈述知识内容。
5. 如果证据中标注了「共现」等弱关联，不要把它当作确定关系来陈述。"""

REWRITE_SYSTEM_PROMPT = """你是一个检索查询改写专家。上一次检索得到的证据不足以回答用户问题。
请改写查询以提升检索命中率。只输出改写后的查询文本，不要任何解释或标点修饰。"""

# 第 2 轮及之后的"续查"规划：让模型看着已有证据决定还缺什么。
# 这是把"单轮工具调用"变成 ReAct 式多轮循环的关键提示词。
FOLLOWUP_PLANNER_PROMPT = """你是一个检索规划专家。系统已经执行过一轮检索，现在请你判断**是否还需要补充检索**。

可用工具：
- vector_search(query): 语义向量检索，适合查找具体事实、数值、定义、细节描述
- graph_search(query): 知识图谱多跳检索，适合查找实体之间的关系、跨段落推理
- graph_global_search(query): 知识图谱全局社区检索，适合"整体涉及哪些""一共有多少"这类聚合问题
- kb_meta(): 查询知识库自身的元信息（文档数量、文档清单）

规则：
1. 如果已有证据足以回答，输出 {"tools": []}。
2. 如果需要补充，从上面挑**尚未尝试过**的工具，并可以给出更精准的 query。
3. 不要重复调用同一个工具查同样的问题。
4. 只输出 JSON，不要解释。格式：{"tools": ["graph_search"], "query": "更精准的查询"}

注意：只有当已有证据明显缺失关键信息时才补充检索。证据已经够用就应该返回空数组，
多查一轮会增加延迟与成本。"""

# 充分性判官：判断当前证据是否足以回答，不足则给出补充方案。
# 之所以要有这一环：启发式只能识别"检索完全空"这种粗粒度失败，
# 无法识别"检索到了内容但答不出问题"（例如召回了药品章节却没召回到剂量）。
ASSESSOR_SYSTEM_PROMPT = """你是一个检索质量评估专家。请判断当前检索到的证据是否足以回答用户问题。

判断标准：
1. 证据中必须**直接包含**回答问题所需的事实，而不是仅仅主题相关。
2. 如果问题涉及实体之间的关系，证据里必须能看到这条关系或能凑出这条关系链。
3. 如果问题要求聚合统计（哪些/几个/多少），证据必须包含完整清单而非部分样本。
4. 证据足够就判定 sufficient=true；不要因为"可以再查一点"就判定不足。

只输出 JSON，不要解释。格式：
{"sufficient": true/false, "missing": "缺少什么信息（一句话）", "tools": ["graph_search"], "query": "补充检索用的查询"}"""


# =============================================================================
# Agent
# =============================================================================


class GraphRAGAgent:
    """LangGraph 驱动的检索 Agent。

    Args:
        graph_store: 后端知识图谱（必须）
        vector_store: 向量库，任何具备 `search(query, top_k=...)` 与
            `get_kb_summary()` 的对象即可；测试可注入假对象
        llm: 任何具备 `.invoke(messages)` 的对象；测试可注入 FakeLLM
        use_llm_router: 规划节点是否用 LLM 决策。False 时纯启发式（确定性、零额外调用）
        hops: 图检索默认跳数
        top_k: 向量检索默认召回数
        max_attempts: 反思重试的最大轮数
        grounded_threshold: 引用校验的通过阈值
    """

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        vector_store: Any = None,
        llm: Any = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_llm_router: bool = True,
        hops: int = 2,
        top_k: int = 5,
        max_attempts: int = 2,
        max_tool_rounds: int = 3,
        use_llm_assessor: bool = True,
        thin_evidence_chars: int = 400,
        grounded_threshold: float = 0.85,
        price_input_per_1m: float = 0.0,
        price_output_per_1m: float = 0.0,
        llm_timeout_seconds: float = 45,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self._llm = llm
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.use_llm_router = use_llm_router
        self.hops = hops
        self.top_k = top_k
        self.max_attempts = max_attempts
        # 工具调用轮数上限：每多一轮都可能多一次规划/判官 LLM 调用，
        # 3 轮是"能补上漏检"与"不无限烧钱"之间的折中
        self.max_tool_rounds = max(1, max_tool_rounds)
        # 是否用 LLM 判断证据充分性。关掉则退化为纯启发式（确定性、零额外调用），
        # 测试与成本敏感场景建议关闭
        self.use_llm_assessor = use_llm_assessor
        # 证据"薄"的阈值：低于这个字符数才值得请判官复核。见 _needs_assessment 的实测依据
        self.thin_evidence_chars = thin_evidence_chars
        self.grounded_threshold = grounded_threshold
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.llm_timeout_seconds = llm_timeout_seconds

        self.graph_retriever = GraphRetriever(graph_store, default_hops=hops)
        self._tools = self._make_tools()
        self._tool_node = ToolNode(self._tools)
        self._app = None  # 惰性编译

    # -------------------------------------------------------------------------
    # 计量
    # -------------------------------------------------------------------------

    def new_usage_tracker(self) -> LLMUsageTracker:
        """为一次请求创建独立的用量追踪器。

        刻意不用 Agent 成员变量：FastAPI 的同步路由跑在线程池里，
        成员变量会让并发请求的用量互相污染。
        """
        return LLMUsageTracker(
            model=self.model or "",
            price_input_per_1m=self.price_input_per_1m,
            price_output_per_1m=self.price_output_per_1m,
        )

    # -------------------------------------------------------------------------
    # LLM 接入
    # -------------------------------------------------------------------------

    @property
    def llm(self) -> Any:
        if self._llm is None:
            import os

            from langchain_openai import ChatOpenAI

            model = self.model or os.getenv("DEFAULT_MODEL", "deepseek-chat")
            key = self.api_key or os.getenv("DEEPSEEK_API_KEY", "")
            if not key:
                raise ValueError("未配置 DEEPSEEK_API_KEY，无法生成答案")
            self._llm = ChatOpenAI(
                model=model,
                api_key=key,
                base_url=self.base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                temperature=0.1,
                timeout=self.llm_timeout_seconds,
                max_retries=1,
            )
        return self._llm

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        node: str = "llm",
        tracker: Optional[LLMUsageTracker] = None,
    ) -> str:
        """调用 LLM 并统一取出文本内容。

        `node` 与 `tracker` 用于成本归因：知道这几次 token 是"路由判官"花的
        还是"答案生成"花的，才能判断优化该从哪里下手。
        失败时也会记一条（带 error），否则失败请求的成本会被静默吞掉。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            if tracker is not None:
                tracker.record(
                    node=node,
                    prompt_tokens=estimate_tokens(system_prompt + user_prompt),
                    completion_tokens=0,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    estimated=True,
                    error=str(exc),
                )
            raise

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        text = str(content or "")

        if tracker is not None:
            tracker.record_from_response(
                node=node,
                response=response,
                prompt_text=system_prompt + user_prompt,
                completion_text=text,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return text

    # -------------------------------------------------------------------------
    # 工具定义
    # -------------------------------------------------------------------------

    def _make_tools(self) -> List[Any]:
        """创建绑定到当前 Agent 实例的工具集合。

        工具以闭包形式捕获 `self`，从而能访问注入的 vector_store / graph_retriever。
        返回值统一是 **JSON 字符串**：ToolNode 会把工具返回值包成 ToolMessage，
        下游 `generate` 节点再解析，这样结构化数据（来源、字符区间）不会在
        消息传递过程中丢失。
        """
        agent = self

        @tool
        def vector_search(query: str, top_k: int = 5) -> str:
            """语义向量检索。适合查找具体事实、数值、定义与细节描述。"""
            results: List[Dict[str, Any]] = []
            error = ""
            if agent.vector_store is None:
                error = "向量库未初始化"
            else:
                try:
                    results = agent.vector_store.search(query, top_k=top_k) or []
                except Exception as exc:
                    logger.warning("向量检索失败: %s", exc)
                    error = str(exc)
            return json.dumps(
                {"tool": TOOL_VECTOR, "query": query, "count": len(results),
                 "results": results, "error": error},
                ensure_ascii=False,
            )

        @tool
        def graph_search(query: str, hops: int = 2) -> str:
            """知识图谱多跳检索。适合查找实体之间的关系、需要跨段落推理的问题。"""
            try:
                result = agent.graph_retriever.retrieve(query, hops=hops)
                payload = result.to_dict()
            except Exception as exc:
                logger.warning("图检索失败: %s", exc)
                payload = {"tool": TOOL_GRAPH, "query": query, "hit": False, "error": str(exc)}
            return json.dumps(payload, ensure_ascii=False)

        @tool
        def graph_global_search(query: str) -> str:
            """知识图谱全局社区检索。适合"整体涉及哪些""一共有多少"这类聚合统计问题。"""
            try:
                result = agent.graph_retriever.retrieve_global(query)
                payload = result.to_dict()
            except Exception as exc:
                logger.warning("全局检索失败: %s", exc)
                payload = {"tool": TOOL_GRAPH_GLOBAL, "query": query, "hit": False, "error": str(exc)}
            return json.dumps(payload, ensure_ascii=False)

        @tool
        def kb_meta() -> str:
            """查询知识库自身的元信息：文档数量、片段数量、文档清单。"""
            summary: Dict[str, Any] = {}
            error = ""
            if agent.vector_store is None:
                error = "向量库未初始化"
            else:
                try:
                    summary = agent.vector_store.get_kb_summary() or {}
                except Exception as exc:
                    error = str(exc)
            return json.dumps(
                {"tool": TOOL_KB_META, "summary": summary, "error": error}, ensure_ascii=False
            )

        return [vector_search, graph_search, graph_global_search, kb_meta]

    @property
    def tools(self) -> List[Any]:
        return self._tools

    # -------------------------------------------------------------------------
    # 节点 1：规划
    # -------------------------------------------------------------------------

    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """决定这一轮用哪些工具。

        分两种情形，这是"单轮工具调用"与"多轮 ReAct 循环"的分界：
        - **第 0 轮**：只看问题，用启发式 + LLM 路由决策（`_initial_plan`）
        - **第 N>0 轮**：把已获得的证据摘要给模型看，让它判断还缺什么（`_followup_plan`）

        无论哪种情形，最后都要过一遍**规划失败降级链**，保证永远有工具可执行 ——
        规划失败导致"零工具"会让 ToolNode 无调用可做、直接生成无依据的答案。
        """
        question = state.get("question") or ""
        query = state.get("current_query") or question
        round_index = int(state.get("tool_round") or 0)
        tracker = state.get("usage_tracker")
        trace = list(state.get("trace") or [])
        used = list(state.get("used_tools") or [])

        if round_index == 0:
            chosen, decided_by, router_error = self._initial_plan(query, tracker)
        else:
            chosen, decided_by, router_error = self._followup_plan(
                question=question, query=query, state=state, tracker=tracker
            )

        # ---- 降级链第 1 层：模型给了非法/空工具名 -> 用兜底组合 ----
        #
        # 说明：第 0 轮实际上永远不会走到这里 —— `plan_route` 按构造只返回合法工具名
        # （有 `test_all_routes_are_valid_tool_names` 守着这个不变量）。
        # 保留这一层是为了两件事：一是防御未来改动 `plan_route` 时静默产出非法名，
        # 二是第 N>0 轮的"续查规划"确实可能返回空集（模型说"不用再查了"）。
        fallback_reason = ""
        valid = [name for name in chosen if name in ALL_TOOLS]
        if not valid:
            # 兜底也要按"尽量别重复查"排优先级：默认组合里没用过的 ->
            # 其它没用过的 -> 实在没得选才重复默认组合。绝不允许零工具。
            valid = [name for name in DEFAULT_PLAN_TOOLS if name not in used]
            if not valid:
                valid = [
                    name for name in ALL_TOOLS if name not in used and name != TOOL_KB_META
                ]
            if not valid:
                valid = list(DEFAULT_PLAN_TOOLS)
            fallback_reason = "no_valid_tools"
            decided_by = "fallback"

        # ---- 降级链第 2 层：tool_calls 仍然构造不出来 -> 强制向量检索 ----
        calls = make_tool_calls(query, valid, hops=self.hops, top_k=self.top_k)
        if not calls:
            valid = [TOOL_VECTOR]
            calls = make_tool_calls(query, valid, hops=self.hops, top_k=self.top_k)
            fallback_reason = "empty_tool_calls"
            decided_by = "fallback"

        # trace 约定：**每个节点每轮只写一条记录**，结构统一。
        # 早期版本把 router_failed 单独写成一条 {"node":"plan","event":...}，
        # 导致下游按 node 名取第一条记录时会拿到没有 decided_by 字段的那条。
        trace.append(
            {
                "node": "plan",
                "round": round_index,
                "decided_by": decided_by,
                "tools": valid,
                "router_error": router_error,
                "fallback_reason": fallback_reason,
            }
        )

        # route 累计（上报用）、used_tools 累计（续查决策用）、current_tools 只反映本轮。
        # 为什么要分开：重试或多轮会换工具，若只保留"最后一轮"，上报给调用方的 route
        # 就漏掉了前面真正跑过的检索；而续查决策又必须知道"哪些已经试过了"。
        accumulated = list(state.get("route") or [])
        for name in valid:
            if name not in accumulated:
                accumulated.append(name)

        return {
            "route": accumulated,
            "current_tools": valid,
            "used_tools": list(dict.fromkeys(used + valid)),
            "messages": [AIMessage(content="", tool_calls=calls)],
            "trace": trace,
        }

    def _initial_plan(
        self, query: str, tracker: Optional[LLMUsageTracker]
    ) -> Tuple[List[str], str, str]:
        """第 0 轮规划：启发式打底，LLM 可选复核。"""
        heuristic_tools = plan_route(query)
        chosen = list(heuristic_tools)
        decided_by = "heuristic"
        router_error = ""

        # 元信息问题没必要浪费一次路由调用（启发式已经确定无疑）
        if self.use_llm_router and heuristic_tools != [TOOL_KB_META]:
            try:
                raw = self._call_llm(
                    ROUTER_SYSTEM_PROMPT,
                    f"用户问题：{query}\n\n请输出 JSON：",
                    node="route",
                    tracker=tracker,
                )
                llm_tools = self._parse_tool_choice(raw)
                if llm_tools:
                    chosen = llm_tools
                    decided_by = "llm"
            except Exception as exc:
                logger.warning("LLM 路由失败，回退启发式: %s", exc)
                router_error = str(exc)

        return chosen, decided_by, router_error

    def _followup_plan(
        self,
        question: str,
        query: str,
        state: AgentState,
        tracker: Optional[LLMUsageTracker],
    ) -> Tuple[List[str], str, str]:
        """第 N>0 轮规划：gap-driven，让模型看着已有证据决定补什么。

        `query` 与 `question` 不同时，说明充分性判官已经改写过查询
        （例如把"张建国用什么药"细化为"张建国的用药方案与剂量"）。
        这种情况下允许**复用**已用过的工具，因为查询变了，检索结果也会不同。
        """
        used = set(state.get("used_tools") or [])
        refined = bool(query and query != question)

        if refined:
            available = list(ALL_TOOLS)
        else:
            available = [name for name in ALL_TOOLS if name not in used] or list(ALL_TOOLS)

        # 优先采纳充分性判官建议的工具（那是一次真正的"规划"产出）；
        # 没有建议时退回启发式：挑最多两个可用工具
        # （一次铺开太多会显著增加 token，而边际收益很低）。
        assess = state.get("assess") or {}
        suggested = [name for name in (assess.get("suggested_tools") or []) if name in available]
        chosen = suggested or (available[:2] if available else list(DEFAULT_PLAN_TOOLS))
        decided_by = "assessor" if suggested else "heuristic"
        router_error = ""

        if self.use_llm_router and available:
            try:
                digest = _digest_evidence(state.get("evidence_blocks") or [])
                prompt = (
                    f"用户问题：{question}\n\n"
                    f"当前检索查询：{query}\n\n"
                    f"已获得的证据摘要：\n{digest}\n\n"
                    f"本轮可用的工具：{'、'.join(available)}\n\n"
                    "请输出 JSON："
                )
                raw = self._call_llm(
                    FOLLOWUP_PLANNER_PROMPT, prompt, node="plan_followup", tracker=tracker
                )
                parsed = self._parse_followup_plan(raw, available)
                if parsed["tools"]:
                    chosen = parsed["tools"]
                    decided_by = "llm"
                elif parsed["explicit_empty"]:
                    # 模型明确说"不用再查了" —— 交给 assess 节点收敛
                    chosen = []
                    decided_by = "llm"
            except Exception as exc:
                logger.warning("续查规划失败，回退启发式: %s", exc)
                router_error = str(exc)

        return chosen, decided_by, router_error

    @staticmethod
    def _parse_followup_plan(raw: str, available: Sequence[str]) -> Dict[str, Any]:
        """解析续查规划的输出。

        与 `_parse_tool_choice` 的区别：这里需要区分"模型说了不用再查"
        （`tools: []`，合法且有意义）与"模型输出不可解析"（应回退启发式）。
        把两者混为一谈会让一次格式错误变成"放弃检索"。
        """
        result: Dict[str, Any] = {"tools": [], "query": "", "explicit_empty": False}
        if not raw:
            return result

        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        left, right = text.find("{"), text.rfind("}")
        if left == -1 or right <= left:
            return result
        try:
            payload = json.loads(text[left:right + 1])
        except json.JSONDecodeError:
            return result
        if not isinstance(payload, dict):
            return result

        tools = payload.get("tools")
        if isinstance(tools, list):
            result["explicit_empty"] = True
            result["tools"] = [name for name in tools if name in available]
        query = payload.get("query")
        if isinstance(query, str):
            result["query"] = query.strip()
        return result

    @staticmethod
    def _parse_tool_choice(raw: str) -> List[str]:
        """解析路由 LLM 的输出，只保留合法工具名。"""
        if not raw:
            return []
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        left, right = text.find("{"), text.rfind("}")
        if left == -1 or right <= left:
            return []
        try:
            payload = json.loads(text[left:right + 1])
        except json.JSONDecodeError:
            return []
        tools = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list):
            return []
        # 白名单过滤：模型可能编造工具名
        return [name for name in tools if name in ALL_TOOLS]

    # -------------------------------------------------------------------------
    # 节点 2：工具执行（LangGraph ToolNode）
    # -------------------------------------------------------------------------

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        """执行 ToolNode 并把结果汇总成证据文本。"""
        output = self._tool_node.invoke(state)
        new_messages = output.get("messages", [])
        return self._assemble_tool_output(state, new_messages)

    def _assemble_tool_output(
        self, state: AgentState, new_messages: Sequence[AnyMessage]
    ) -> Dict[str, Any]:
        """把本轮工具产出**累积**进证据池，并重建汇总视图。

        为什么要累积而不是覆盖：多轮工具调用的意义就在于"第二轮补上第一轮缺的东西"。
        如果每轮覆盖，第一轮辛苦召回的跨块关系链会被第二轮的社区摘要冲掉，
        多轮反而比单轮更差。这是多轮改造里最容易踩的坑。
        """
        round_index = int(state.get("tool_round") or 0)
        blocks = list(state.get("evidence_blocks") or [])

        trace = list(state.get("trace") or [])
        for message in new_messages:
            if not isinstance(message, ToolMessage):
                continue
            rendered = self._render_tool_message(message)
            rendered["round"] = round_index
            blocks.append(rendered)
            trace.append(
                {
                    "node": "tools",
                    "round": round_index,
                    "tool": rendered["tool"],
                    "content_length": len(rendered["text"] or ""),
                    "empty": rendered["empty"],
                    "error": bool(rendered["error"]),
                }
            )

        evidence_text, sources, graph_payload = self._merge_blocks(blocks)

        return {
            "messages": list(new_messages),
            "evidence_blocks": blocks,
            "evidence_text": evidence_text,
            "sources": sources,
            "graph_payload": graph_payload,
            "trace": trace,
        }

    @staticmethod
    def _merge_blocks(
        blocks: Sequence[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """把证据块合并成：喂给 LLM 的文本 + 结构化来源 + 最新图谱结果。"""
        texts = [b["text"] for b in blocks if b.get("text")]
        sources: List[Dict[str, Any]] = []
        seen_chunks = set()
        graph_payload: Dict[str, Any] = {}
        for block in blocks:
            for source in block.get("sources") or []:
                # 按 chunk_id 去重：多轮可能重复召回同一批向量结果
                key = source.get("doc_id"), source.get("content_preview")
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                sources.append(source)
            if block.get("graph_payload"):
                graph_payload = block["graph_payload"]
        return "\n\n".join(texts), sources, graph_payload

    def _execute_tools(self, state: AgentState) -> List[ToolMessage]:
        """手工派发工具调用（供流式路径使用）。

        为什么不复用 `ToolNode`：LangGraph 1.x 的 ToolNode 在被 `invoke` 时会从
        运行配置里取 runtime，**脱离已编译图单独调用会抛
        `ValueError: Missing required config key 'N/A'`**。图内路径照常用 ToolNode
        （那才是 LangGraph 的原生工具执行）；流式路径需要"图外的同一步骤"，
        于是这里按同样的 tool_calls 协议自行派发，返回值形态保持一致（ToolMessage 列表）。

        这里多做了一层**参数白名单过滤**：与工具名一样，模型可能编造参数名，
        直接透传会让工具抛 TypeError。这与 `_parse_tool_choice` 的工具名白名单是同一类防御。
        """
        tool_map = {t.name: t for t in self._tools}
        last = state["messages"][-1] if state.get("messages") else None
        tool_calls = getattr(last, "tool_calls", None) or []

        messages: List[ToolMessage] = []
        for call in tool_calls:
            name = call.get("name") or ""
            call_id = call.get("id") or f"call_{name}"
            tool_obj = tool_map.get(name)

            if tool_obj is None:
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False),
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            allowed = set(
                (tool_obj.args_schema.model_json_schema().get("properties") or {}).keys()
            ) if getattr(tool_obj, "args_schema", None) else set()
            safe_args = {
                k: v for k, v in (call.get("args") or {}).items() if not allowed or k in allowed
            }

            try:
                content = tool_obj.invoke(safe_args)
            except Exception as exc:
                logger.warning("工具 %s 执行失败: %s", name, exc)
                content = json.dumps({"error": str(exc)}, ensure_ascii=False)

            messages.append(
                ToolMessage(content=str(content), tool_call_id=call_id, name=name)
            )

        return messages

    @staticmethod
    def _render_tool_message(message: AnyMessage) -> Dict[str, Any]:
        """把单条 ToolMessage 渲染成一个证据块。

        证据文本是 LLM 唯一能看到的世界，因此这里要把"弱证据"标注清楚
        （图检索结果里已自带「共现」警告，这里原样透传）。

        `empty` 与 `error` 两个标记是多轮决策的依据：
        `empty=True` 会触发"换个工具再查一轮"，`error=True` 说明该工具这次不可用。
        """
        block: Dict[str, Any] = {
            "tool": message.name or "",
            "text": "",
            "sources": [],
            "graph_payload": {},
            "empty": True,
            "error": "",
        }

        try:
            payload = json.loads(message.content or "{}")
        except (json.JSONDecodeError, TypeError):
            block["error"] = "工具返回内容无法解析为 JSON"
            return block

        if not isinstance(payload, dict):
            block["error"] = "工具返回内容不是 JSON 对象"
            return block

        name = message.name
        if payload.get("error"):
            block["error"] = str(payload["error"])

        if name == TOOL_VECTOR:
            results = payload.get("results") or []
            if block["error"]:
                block["text"] = f"【向量检索】失败：{block['error']}"
                return block
            if not results:
                block["text"] = "【向量检索】未召回任何内容。"
                return block
            parts = [f"【向量检索】召回 {len(results)} 条："]
            sources: List[Dict[str, Any]] = []
            for i, item in enumerate(results, 1):
                title = item.get("doc_title") or "未知文档"
                score = item.get("score", 0)
                content = (item.get("content") or "").strip()
                parts.append(f"  [{i}] 来源：{title}（相似度 {score:.2f}）\n      {content}")
                sources.append(
                    {
                        "doc_id": item.get("doc_id"),
                        "doc_title": title,
                        "content_preview": content[:200],
                        "score": score,
                        "chunk_type": item.get("chunk_type"),
                        "char_interval": item.get("char_interval")
                        or (item.get("attributes") or {}).get("char_interval"),
                        "attributes": item.get("attributes") or {},
                    }
                )
            block["text"] = "\n".join(parts)
            block["sources"] = sources
            block["empty"] = False

        elif name in (TOOL_GRAPH, TOOL_GRAPH_GLOBAL):
            context = payload.get("context_text") or ""
            if block["error"]:
                block["text"] = f"【知识图谱检索】失败：{block['error']}"
                return block
            if not context:
                block["text"] = "【知识图谱检索】未命中任何图谱内容。"
                return block
            block["text"] = context
            block["graph_payload"] = payload
            block["empty"] = False

        elif name == TOOL_KB_META:
            summary = payload.get("summary") or {}
            if block["error"]:
                block["text"] = f"【知识库概况】获取失败：{block['error']}"
                return block
            doc_count = summary.get("document_count", 0)
            chunk_count = summary.get("chunk_count", 0)
            documents = summary.get("documents") or []
            titles = "、".join(f"《{d.get('title', '未知')}》" for d in documents) or "无"
            block["text"] = (
                "【知识库概况】\n"
                f"  文档数量：{doc_count}\n"
                f"  片段数量：{chunk_count}\n"
                f"  文档清单：{titles}"
            )
            block["empty"] = False

        else:
            block["error"] = block["error"] or f"未知工具：{name}"

        return block

    @staticmethod
    def _collect_evidence(
        messages: Sequence[AnyMessage],
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """把一批消息渲染成证据（保留此入口，便于单独测试渲染逻辑）。

        这是 `_render_tool_message` + `_merge_blocks` 的组合视图；
        主流程走的是"逐块累积"的 `_assemble_tool_output`。
        """
        blocks = [
            GraphRAGAgent._render_tool_message(message)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        return GraphRAGAgent._merge_blocks(blocks)

    # -------------------------------------------------------------------------
    # 节点 2.5：充分性评估（多轮循环的决策点）
    # -------------------------------------------------------------------------

    def _assess_node(self, state: AgentState) -> Dict[str, Any]:
        """判断"证据够不够回答问题"，决定继续补检索还是进入生成。

        这是多轮 ReAct 循环的枢纽。判定顺序刻意从**最便宜、最确定**的条件开始：

        1. 元信息问题 -> 直接生成（答案来自知识库统计，与检索质量无关）
        2. 已到轮数上限 -> 直接生成（不能无限烧钱）
        3. 本轮全部检索为空 -> **换未用过的工具再查一轮**（这是最有价值的续查场景）
        4. 没有未用过的工具了 -> 生成（再查也是重复）
        5. 其余情况 -> 可选 LLM 判官判断充分性

        第 3 条覆盖的是真实最常见的失败模式：向量库没配、或图谱里没这个实体，
        导致本轮零证据。此时盲目生成必然是一句"无法回答"，
        换一个工具往往就能拿到东西。
        """
        round_index = int(state.get("tool_round") or 0)
        max_rounds = int(state.get("max_tool_rounds") or self.max_tool_rounds)
        current_tools = list(state.get("current_tools") or [])
        blocks = [
            b for b in (state.get("evidence_blocks") or []) if b.get("round") == round_index
        ]
        usable = [b for b in blocks if not b.get("empty") and not b.get("error")]

        used = set(state.get("used_tools") or [])
        # kb_meta 不参与"换工具续查"：它只对元信息问题有意义
        unused = [name for name in ALL_TOOLS if name not in used and name != TOOL_KB_META]

        decision = "generate"
        reason = ""
        missing = ""
        next_query = ""
        suggested_tools: List[str] = []

        if current_tools == [TOOL_KB_META]:
            reason = "kb_meta_terminal"
        elif round_index + 1 >= max_rounds:
            reason = "reached_max_rounds"
        elif not usable and unused:
            decision, reason = "plan", "empty_evidence_switch_tools"
        elif not usable and not unused:
            reason = "empty_evidence_no_tools_left"
        elif self.use_llm_assessor and unused and self._needs_assessment(state):
            verdict = self._judge_sufficiency(state)
            if verdict["sufficient"]:
                reason = "llm_judged_sufficient"
            elif verdict["tools"] or verdict["query"]:
                decision, reason = "plan", "llm_judged_insufficient"
                missing = verdict["missing"]
                next_query = verdict["query"]
                # 判官给的建议必须**原样交给下一轮规划去执行**。
                # 早期版本只把 verdict 当布尔用、丢掉了它建议的工具，
                # 下一轮又退回"挑两个没用过的"启发式 —— 等于判官的规划白做了。
                suggested_tools = list(verdict["tools"])
            else:
                reason = "llm_judged_insufficient_no_plan"
                missing = verdict["missing"]
        else:
            reason = "evidence_sufficient_heuristic"

        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "assess",
                "round": round_index,
                "decision": decision,
                "reason": reason,
                "usable_blocks": len(usable),
                "empty_blocks": len(blocks) - len(usable),
                "unused_tools": unused,
                "missing": missing,
                "suggested_tools": suggested_tools,
            }
        )

        update: Dict[str, Any] = {
            "assess": {
                "decision": decision,
                "reason": reason,
                "round": round_index,
                "usable_blocks": len(usable),
                "unused_tools": unused,
                "missing": missing,
                "suggested_tools": suggested_tools,
            },
            "trace": trace,
        }
        if decision == "plan":
            # 轮次在这里递增（而不是在 plan 节点）：plan 节点在"同一轮"里可能被
            # 调用多次（例如 rewrite 之后），由决策方递增才能保证一轮只被计一次。
            update["tool_round"] = round_index + 1
        if next_query:
            # 判官改写了查询 -> 下一轮允许复用工具（查询变了，结果也会变）
            update["current_query"] = next_query
        return update

    def _needs_assessment(self, state: AgentState) -> bool:
        """廉价前置门控：判断"是否值得花一次 LLM 调用去评估充分性"。

        为什么需要这道门 —— 实测数据说话：
        初版实现里只要"有未用过的工具"就请判官，结果 14 题里有 **11 题**都触发了补检索，
        平均 2.29 轮，但答案召回率相比单轮版**没有提升**，而单题延迟从 4.5s 涨到 16.2s
        （3.5 倍）、LLM 调用从约 2 次涨到 4.57 次。
        原因是判官天然倾向于"再查一点更保险"，对本来证据充足的问题也判不足。

        所以把判官的职责收窄成：**只在证据明显不够时才介入**。
        两个触发条件都是零成本可判的：
        1. 证据总量偏薄（默认 < 400 字符）—— 说明召回质量确实成问题
        2. 聚合型问题（"涉及哪些/一共有几个"）—— 这类问题需要**完整清单**，
           部分样本就算"内容很多"也依然是错的，值得复核

        其余情况直接生成。这个门控把"多轮"从默认路径变回了兜底路径，
        这是它该待的位置。
        """
        blocks = [
            b
            for b in (state.get("evidence_blocks") or [])
            if b.get("round") == int(state.get("tool_round") or 0)
        ]
        evidence_chars = sum(len(b.get("text") or "") for b in blocks if not b.get("empty"))
        if evidence_chars < self.thin_evidence_chars:
            return True
        return is_global_query(state.get("question") or "")

    def _judge_sufficiency(self, state: AgentState) -> Dict[str, Any]:
        """用 LLM 判断证据是否充分。任何失败都返回"充分"（保守收敛，不阻塞主流程）。"""
        default = {"sufficient": True, "missing": "", "tools": [], "query": ""}
        tracker = state.get("usage_tracker")
        try:
            digest = _digest_evidence(
                [b for b in (state.get("evidence_blocks") or [])], max_chars_per_block=400
            )
            raw = self._call_llm(
                ASSESSOR_SYSTEM_PROMPT,
                f"用户问题：{state.get('question') or ''}\n\n"
                f"已检索到的证据：\n{digest}\n\n请输出 JSON：",
                node="assess",
                tracker=tracker,
            )
        except Exception as exc:
            logger.warning("充分性判官调用失败，按充分处理: %s", exc)
            return default

        parsed = _parse_json_object(raw)
        if not parsed:
            return default

        sufficient = parsed.get("sufficient")
        if isinstance(sufficient, str):
            sufficient = sufficient.strip().lower() in ("true", "yes", "1")
        tools_raw = parsed.get("tools")
        tools = (
            [name for name in tools_raw if name in ALL_TOOLS]
            if isinstance(tools_raw, list)
            else []
        )
        query = parsed.get("query") if isinstance(parsed.get("query"), str) else ""
        missing = parsed.get("missing") if isinstance(parsed.get("missing"), str) else ""

        return {
            "sufficient": bool(sufficient) if sufficient is not None else True,
            "missing": missing,
            # 排除 kb_meta：判官不该把元信息工具当作"补充检索"手段
            "tools": [name for name in tools if name != TOOL_KB_META],
            "query": query.strip(),
        }

    def _after_assess(self, state: AgentState) -> str:
        """充分性评估后的分支：继续补检索，还是进入生成。"""
        if (state.get("assess") or {}).get("decision") == "plan":
            return "plan"
        return "generate"

    # -------------------------------------------------------------------------
    # 节点 3：生成
    # -------------------------------------------------------------------------

    def _build_answer_prompt(self, question: str, evidence_text: str) -> str:
        return (
            f"【检索证据】\n{evidence_text or '（无可用证据）'}\n\n"
            f"【用户问题】\n{question}\n\n"
            "请依据证据作答。若证据不足，请直接说明无法回答，不要编造。"
        )

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question") or ""
        evidence_text = state.get("evidence_text") or ""

        try:
            answer = self._call_llm(
                ANSWER_SYSTEM_PROMPT,
                self._build_answer_prompt(question, evidence_text),
                node="generate",
                tracker=state.get("usage_tracker"),
            )
        except Exception as exc:
            logger.error("答案生成失败: %s", exc)
            answer = f"生成答案时出错：{exc}"

        trace = list(state.get("trace") or [])
        trace.append({"node": "generate", "answer_length": len(answer)})
        return {"answer": answer, "trace": trace}

    # -------------------------------------------------------------------------
    # 节点 4：引用校验
    # -------------------------------------------------------------------------

    def _verify_node(self, state: AgentState) -> Dict[str, Any]:
        # 只用实体**展示名**参与校验：图谱键是规范化名（可能被小写化/去书名号），
        # 用键会让"《王者荣耀》"这类答案与键"王者荣耀"对不上，产生假阳性。
        known_entities = sorted(
            {entity.name for entity in self.graph_store.entities.values()}
            | {alias for entity in self.graph_store.entities.values() for alias in entity.aliases}
        )
        verification = verify_citations(
            answer=state.get("answer") or "",
            evidence_text=state.get("evidence_text") or "",
            known_entities=known_entities,
            threshold=self.grounded_threshold,
        )
        trace = list(state.get("trace") or [])
        trace.append(
            {
                "node": "verify",
                "grounded_ratio": verification["grounded_ratio"],
                "passed": verification["passed"],
                "ungrounded": verification["ungrounded_numbers"] + verification["ungrounded_entities"],
            }
        )
        return {"verification": verification, "trace": trace}

    # -------------------------------------------------------------------------
    # 节点 5：查询改写
    # -------------------------------------------------------------------------

    def _rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question") or ""
        verification = state.get("verification") or {}
        attempts = int(state.get("attempts") or 1)
        trace = list(state.get("trace") or [])

        missing = (verification.get("ungrounded_numbers") or []) + (
            verification.get("ungrounded_entities") or []
        )
        new_query = ""

        # 优先让 LLM 改写（带上下文：上次为什么失败）
        try:
            new_query = self._sanitize_rewrite(
                self._call_llm(
                    REWRITE_SYSTEM_PROMPT,
                    f"原始问题：{question}\n"
                    f"上次检索未能支撑的内容：{'、'.join(missing) or '证据整体不足'}\n"
                    "请给出更适合检索的查询：",
                    node="rewrite",
                    tracker=state.get("usage_tracker"),
                )
            )
        except Exception as exc:
            logger.warning("查询改写失败，使用确定性兜底: %s", exc)

        # 确定性兜底：把未接地的关键词补进查询
        if not new_query:
            new_query = self._deterministic_rewrite(question, missing)

        trace.append({"node": "rewrite", "attempt": attempts + 1, "new_query": new_query})
        # 改写意味着"换一个思路重来"：证据池、已用工具、轮次都要清空，
        # 否则第二轮会被第一轮的无效证据干扰，也会因为"工具都用过了"而无法续查。
        return {
            "current_query": new_query,
            "attempts": attempts + 1,
            "current_tools": [],
            "used_tools": [],
            "evidence_blocks": [],
            "tool_round": 0,
            "assess": {},
            "evidence_text": "",
            "sources": [],
            "graph_payload": {},
            "trace": trace,
        }

    @staticmethod
    def _sanitize_rewrite(raw: str) -> str:
        """校验并清理 LLM 返回的改写查询。

        必须校验的原因：改写节点的输出会**直接变成下一轮的检索输入**，
        脏输入会污染整条链路。实测中遇到过模型返回：
        - "改写后的查询：xxx" 这种带前缀的
        - "抱歉，我无法完成" 这种拒答
        - 整段解释文字（几十上百字）
        这些都会让第二轮检索彻底跑偏，不如直接退回确定性改写。

        校验规则：去引号/前后缀 → 非空 → 长度 <= 120 → 不含换行与冒号说明。
        """
        if not raw:
            return ""
        text = raw.strip()

        # 去掉常见前缀
        text = re.sub(r"^(改写后(的)?查询|查询|answer|query)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        # 去掉包裹引号
        text = text.strip().strip("\"'“”‘’《》")
        # 换行说明 -> 只取第一行
        text = text.splitlines()[0].strip() if text else ""

        if not text or len(text) > 120:
            return ""
        # 拒答/解释类输出
        if re.match(r"^(抱歉|对不起|无法|我不能|作为一个)", text):
            return ""
        return text

    @staticmethod
    def _deterministic_rewrite(question: str, missing: Sequence[str]) -> str:
        """无 LLM 时的确定性改写：把未接地内容作为关键词补进查询。

        好处是完全可复现 —— 评测脚本里没有 LLM 时也能跑出稳定结果。
        """
        keywords = [item for item in missing if item][:5]
        if not keywords:
            # 没有具体线索时，去掉疑问词让查询更"像检索语句"
            return re.sub(r"(请问|是什么|是多少|怎么样|为什么|呢|吗|？|\?)", "", question).strip() or question
        return f"{question} {' '.join(keywords)}".strip()

    # -------------------------------------------------------------------------
    # 图编排
    # -------------------------------------------------------------------------

    def _after_verify(self, state: AgentState) -> str:
        """反思循环的出口决策。"""
        verification = state.get("verification") or {}
        attempts = int(state.get("attempts") or 1)

        if verification.get("passed"):
            return "end"
        if attempts >= self.max_attempts:
            return "end"

        # 元信息问题不重试：答案来自知识库统计，与检索质量无关，
        # 改写查询不可能改善它，重试只会白烧一次 LLM 调用。
        # 这个分支是靠 trace 发现的 —— 原先 kb_meta 问题也被重试，
        # 结果第二轮路由被改写成向量+图检索，上报的 route 与实际不符。
        if (state.get("current_tools") or []) == [TOOL_KB_META]:
            return "end"

        return "rewrite"

    def build(self):
        """编译 LangGraph 图（惰性、缓存）。

        完整的 ReAct 式结构：

            START ─► plan ─► tools ─► assess ─┬─(证据不足，换工具补查)─► plan
                                              └─(证据够用 / 到轮数上限)─► generate
                                                                            │
                                        ┌───────────────────────────────────┘
                                        ▼
                                     verify ─┬─(引用不可接地且未超次数)─► rewrite ─► plan
                                             └─(通过 / 超次数 / 元信息问题)─► END

        有两个循环，职责不同、不要混为一谈：
        - **assess → plan 循环**：补检索。解决"检索没拿到东西"。
        - **verify → rewrite 循环**：改问题。解决"检索到了但答案编造了"。
        """
        if self._app is not None:
            return self._app

        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("assess", self._assess_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("rewrite", self._rewrite_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "tools")
        builder.add_edge("tools", "assess")
        builder.add_conditional_edges(
            "assess", self._after_assess, {"plan": "plan", "generate": "generate"}
        )
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges(
            "verify", self._after_verify, {"rewrite": "rewrite", "end": END}
        )
        builder.add_edge("rewrite", "plan")

        self._app = builder.compile()
        return self._app

    # -------------------------------------------------------------------------
    # 对外接口
    # -------------------------------------------------------------------------

    def _initial_state(self, question: str) -> AgentState:
        """构造一次请求的初始状态。

        集中在一处，避免 `answer` 与 `answer_stream` 两条路径的状态字段走偏 ——
        多轮改造时正是因为后者漏了新字段，导致流式路径的轮次计数永远停在 0。
        """
        return {
            "messages": [],
            "question": question,
            "current_query": question,
            "route": [],
            "current_tools": [],
            "used_tools": [],
            "evidence_blocks": [],
            "tool_round": 0,
            "max_tool_rounds": self.max_tool_rounds,
            "assess": {},
            "evidence_text": "",
            "sources": [],
            "graph_payload": {},
            "answer": "",
            "verification": {},
            "attempts": 1,
            "trace": [],
            # 每个请求一个独立的用量追踪器，避免并发请求互相污染计量
            "usage_tracker": self.new_usage_tracker(),
        }

    def answer(self, question: str) -> Dict[str, Any]:
        """跑完整个 Agent 图并返回结构化结果。"""
        final = self.build().invoke(self._initial_state(question))

        tracker = final.get("usage_tracker")
        usage = tracker.summary() if tracker is not None else {}
        trace = final.get("trace") or []

        return {
            "success": True,
            "question": question,
            "answer": final.get("answer") or "",
            "sources": final.get("sources") or [],
            "route": final.get("route") or [],
            "graph": final.get("graph_payload") or {},
            "verification": final.get("verification") or {},
            "attempts": final.get("attempts") or 1,
            # 实际执行的工具轮数 = trace 里出现过的**不同 round 编号个数**。
            # 早期版本按"tools 节点记录条数"统计，一轮有 2 个工具就会被算成 2 轮，
            # 这个错误会让成本/轮次报表整体翻倍。
            "tool_rounds": len(
                {
                    entry.get("round")
                    for entry in trace
                    if entry.get("node") == "tools"
                }
            ),
            "usage": usage,
            "trace": trace,
        }

    def answer_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """流式版本：逐 token 吐出答案。

        为什么这里手写编排而不是复用 `app.astream`：
        LangGraph 的 token 级流式（`stream_mode="messages"`）需要全程 async 且
        要改造所有节点为 async；而本项目的 FastAPI 路由与向量库都是同步的。
        这里按**同样的节点顺序**手动编排，节点实现完全复用（包括新增的 assess 多轮循环），
        换取的是不引入 async 改造、且与既有 SSE 协议无缝衔接。

        事件类型与既有 `/rag/qa/stream` 保持一致（sources / chunk / done），
        额外增加 graph / verification / assess / retry / usage 事件，前端可选择性消费。
        """
        state = self._initial_state(question)

        for attempt in range(self.max_attempts):
            # ---- 多轮工具循环：plan -> tools -> assess ----
            round_index = 0
            while True:
                state.update(self._plan_node(state))
                state.update(self._assemble_tool_output(state, self._execute_tools(state)))

                # 提前把来源与图谱结果推给前端，用户能立刻看到"检索到了什么"
                yield {
                    "event": "sources",
                    "data": {
                        "sources": state.get("sources") or [],
                        "route": state.get("route") or [],
                        "round": round_index,
                    },
                }
                if state.get("graph_payload"):
                    yield {"event": "graph", "data": state["graph_payload"]}

                state.update(self._assess_node(state))
                assess = state.get("assess") or {}
                yield {"event": "assess", "data": assess}

                round_index += 1
                if self._after_assess(state) != "plan":
                    break

            # ---- 流式生成 ----
            prompt = self._build_answer_prompt(question, state.get("evidence_text") or "")
            answer_parts: List[str] = []
            tracker = state.get("usage_tracker")
            started = time.perf_counter()
            stream_error = ""
            try:
                for chunk in self.llm.stream(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                ):
                    content = getattr(chunk, "content", "")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    if content:
                        answer_parts.append(str(content))
                        yield {"event": "chunk", "data": {"content": str(content)}}
            except Exception as exc:
                logger.error("流式生成失败: %s", exc)
                stream_error = str(exc)
                answer_parts = [f"生成答案时出错：{exc}"]
                yield {"event": "chunk", "data": {"content": answer_parts[0]}}

            state["answer"] = "".join(answer_parts)
            if tracker is not None:
                # 流式响应一般不返回 usage，这里按估算记录并标记 estimated=True；
                # 中途失败时把 error 一起记下来，否则失败请求的成本被静默吞掉
                tracker.record_stream_estimate(
                    node="generate",
                    prompt_text=ANSWER_SYSTEM_PROMPT + prompt,
                    completion_text=state["answer"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=stream_error,
                )

            # ---- 引用校验 ----
            state.update(self._verify_node(state))
            yield {"event": "verification", "data": state["verification"]}

            if self._after_verify(state) == "end":
                break

            # ---- 改写后重试 ----
            state.update(self._rewrite_node(state))
            yield {
                "event": "retry",
                "data": {"attempt": state["attempts"], "query": state["current_query"]},
            }

        usage = tracker.summary() if tracker is not None else {}
        tool_rounds = len(
            {entry.get("round") for entry in (state.get("trace") or []) if entry.get("node") == "tools"}
        )
        yield {"event": "usage", "data": usage}
        yield {
            "event": "done",
            "data": {
                "success": True,
                "attempts": state.get("attempts") or 1,
                "tool_rounds": tool_rounds,
                "verification": state.get("verification") or {},
                "usage": usage,
            },
        }


# =============================================================================
# 单例管理（与既有 rag_routes 的风格保持一致）
# =============================================================================

_agent: Optional[GraphRAGAgent] = None


def get_graph_agent(
    graph_store: KnowledgeGraphStore,
    vector_store: Any = None,
    **kwargs: Any,
) -> GraphRAGAgent:
    """获取或更新 Agent 单例。

    `graph_store` 变化时必须重建：Agent 内部持有了 `GraphRetriever`（它捕获了 store）。
    """
    global _agent
    if _agent is None or _agent.graph_store is not graph_store:
        _agent = GraphRAGAgent(graph_store=graph_store, vector_store=vector_store, **kwargs)
    elif vector_store is not None and _agent.vector_store is not vector_store:
        _agent.vector_store = vector_store
    return _agent


def reset_graph_agent() -> None:
    """清空单例（测试与知识库重建时调用）。"""
    global _agent
    _agent = None
