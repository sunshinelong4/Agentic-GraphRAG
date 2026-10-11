"""知识图谱与 Agentic 问答 API 路由

与既有 `rag_routes.py` 的关系
----------------------------
- `rag_routes.py` 继续提供原来的向量 RAG 接口（`/rag/qa`、`/rag/search` 等），
  保持向后兼容，前端可以完全不改。
- 本模块新增图谱与 Agent 接口。前端只需把问答请求指向 `/rag/agent/qa/stream`，
  就能获得"多跳图检索 + 引用校验 + 反思重试"，而 SSE 事件协议与原来兼容
  （`sources` / `chunk` / `done`），额外多了 `graph` / `verification` / `retry`。

单例与并发
----------
图谱存储是进程内单例 + 文件持久化。`KnowledgeGraphStore` 内部对写操作加了
`RLock`，因此 FastAPI 线程池里的并发请求是安全的。
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.utils.initialization import serialized_initialization
from app.models.schemas import (
    AgentQARequest,
    AgentQAResponse,
    CommunitiesResponse,
    CommunityInfo,
    EntityDetailResponse,
    GraphBuildRequest,
    GraphBuildResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphStatsResponse,
)
from app.services.graph_retriever import GraphRetriever, is_global_query
from app.services.graph_store import KnowledgeGraphStore, normalize_entity_name
from app.services.relation_extractor import RelationExtractor, build_graph_from_extractions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag/graph", tags=["知识图谱"])

# =============================================================================
# 全局单例
# =============================================================================

_graph_store: Optional[KnowledgeGraphStore] = None
_relation_extractor: Optional[RelationExtractor] = None


@serialized_initialization
def get_graph_store(settings: Settings = Depends(get_settings)) -> KnowledgeGraphStore:
    """获取（并首次加载）知识图谱单例。

    首次访问时从磁盘加载；文件不存在则返回空图谱，而不是报错 ——
    这样全新部署不需要任何初始化步骤就能启动服务。
    """
    global _graph_store
    if _graph_store is None:
        _graph_store = KnowledgeGraphStore.load(settings.graph_persist_path)
        stats = _graph_store.stats()
        logger.info(
            "知识图谱已就绪: 实体=%d 关系=%d 文档=%d",
            stats["entity_count"], stats["relation_count"], stats["document_count"],
        )
    return _graph_store


@serialized_initialization
def get_relation_extractor(settings: Settings = Depends(get_settings)) -> RelationExtractor:
    """获取关系抽取器单例。"""
    global _relation_extractor
    if _relation_extractor is None:
        _relation_extractor = RelationExtractor(
            model=settings.default_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _relation_extractor


def reset_graph_store() -> None:
    """清空图谱单例（测试与重建时使用）。"""
    global _graph_store
    _graph_store = None


def _persist(store: KnowledgeGraphStore, settings: Settings) -> str:
    """落盘，失败只告警不影响请求结果。"""
    try:
        return store.save(settings.graph_persist_path)
    except Exception as exc:
        logger.warning("图谱持久化失败（不影响本次请求）: %s", exc)
        return ""


def _get_vector_store(settings: Settings):
    """复用 rag_routes 的向量库单例，避免同一进程里建两份 Chroma 客户端。"""
    from app.api.rag_routes import get_vector_store

    try:
        return get_vector_store(settings)
    except HTTPException:
        # 未配置 DASHSCOPE_API_KEY 时图谱功能仍应可用，只是没有向量检索
        logger.warning("向量库不可用，Agent 将只有图谱检索能力")
        return None
    except Exception as exc:
        logger.warning("初始化向量库失败: %s", exc)
        return None


# =============================================================================
# 图谱构建
# =============================================================================

@router.post("/build", response_model=GraphBuildResponse, summary="从抽取结果构建知识图谱")
def build_graph(
    request: GraphBuildRequest,
    settings: Settings = Depends(get_settings),
):
    """把一次文档抽取的结果落成图谱。

    完整链路：
        实体入库（带 char_interval，可溯源）
          -> LLM 关系抽取（实体锚定 + 证据必填 + 关系词规范）
          -> 关系入库（拒绝自环与悬空端点）
          -> 持久化

    `build_relations=False` 时只建实体节点，不调用 LLM —— 适合离线导入。
    """
    try:
        store = get_graph_store(settings)
        doc_id = request.doc_id or str(uuid.uuid4())

        extractions = [
            {
                "extraction_class": ext.extraction_class,
                "extraction_text": ext.extraction_text,
                "attributes": ext.attributes or {},
                "char_interval": (
                    {"start_pos": ext.char_interval.start_pos, "end_pos": ext.char_interval.end_pos}
                    if ext.char_interval
                    else None
                ),
            }
            for ext in request.extractions
        ]

        if not request.build_relations:
            entity_stats = store.add_extractions(doc_id, request.doc_title, extractions)
            _persist(store, settings)
            return GraphBuildResponse(
                success=True,
                doc_id=doc_id,
                graph_stats=store.stats(),
                relation_extraction={"skipped": "build_relations=False"},
                **entity_stats,
            )

        extractor = get_relation_extractor(settings)
        result = build_graph_from_extractions(
            store=store,
            extractor=extractor,
            text=request.text,
            doc_id=doc_id,
            doc_title=request.doc_title,
            extractions=extractions,
            max_relations=request.max_relations,
        )
        _persist(store, settings)

        return GraphBuildResponse(
            success=True,
            doc_id=result["doc_id"],
            entities_added=result["entities_added"],
            entities_merged=result["entities_merged"],
            relations_added=result["relations_added"],
            relations_merged=result["relations_merged"],
            rejected=result["rejected"],
            graph_stats=result["graph_stats"],
            relation_extraction=result["relation_extraction"],
        )

    except Exception as exc:
        logger.error("构建知识图谱失败: %s", exc, exc_info=True)
        return GraphBuildResponse(success=False, error=str(exc))


# =============================================================================
# 统计与社区
# =============================================================================

@router.get("/stats", response_model=GraphStatsResponse, summary="获取知识图谱统计")
def graph_stats(settings: Settings = Depends(get_settings)):
    """图谱规模、溯源率、社区数与模块度。

    这些指标既是运维监控，也是**评测与简历上的数字来源**：
    - `traceable_ratio` 反映抽取质量（能否定位回原文）
    - `modularity` 反映社区结构是否显著（>0.3 通常认为有意义）
    """
    try:
        store = get_graph_store(settings)
        stats = store.stats()
        communities = store.detect_communities() if store.entities else []
        modularity = store.modularity(communities) if communities else 0.0

        return GraphStatsResponse(
            success=True,
            entity_count=stats["entity_count"],
            relation_count=stats["relation_count"],
            document_count=stats["document_count"],
            mention_count=stats["mention_count"],
            traceable_ratio=stats["traceable_ratio"],
            avg_degree=stats["avg_degree"],
            community_count=len(communities),
            modularity=round(modularity, 4),
            entity_type_distribution=stats["entity_type_distribution"],
            persist_path=settings.graph_persist_path,
        )
    except Exception as exc:
        logger.error("获取图谱统计失败: %s", exc)
        return GraphStatsResponse(success=False, error=str(exc))


@router.get("/communities", response_model=CommunitiesResponse, summary="获取知识图谱社区划分")
def graph_communities(settings: Settings = Depends(get_settings)):
    """社区列表 + 模块度。

    这是 GraphRAG **global search** 的基础：聚合型问题（"整体涉及哪些机构"）
    的答案不在任何单个文本块里，只能靠社区摘要给出。
    """
    try:
        store = KnowledgeGraphStore.from_dict(get_graph_store(settings).to_dict())
        retriever = GraphRetriever(store)
        communities = store.detect_communities()

        summaries = [retriever.summarize_community(community) for community in communities]

        return CommunitiesResponse(
            success=True,
            community_count=len(summaries),
            modularity=round(store.modularity(communities), 4) if communities else 0.0,
            communities=[
                CommunityInfo(
                    id=summary["id"],
                    size=summary["size"],
                    entity_type_distribution=summary["entity_type_distribution"],
                    core_entities=summary["core_entities"],
                    relation_types=summary["relation_types"],
                    members=summary["members"],
                )
                for summary in summaries
            ],
        )
    except Exception as exc:
        logger.error("获取社区划分失败: %s", exc)
        return CommunitiesResponse(success=False, error=str(exc))


# =============================================================================
# 图谱检索
# =============================================================================

@router.post("/query", response_model=GraphQueryResponse, summary="图谱多跳/全局检索")
def graph_query(
    request: GraphQueryRequest,
    settings: Settings = Depends(get_settings),
):
    """执行图谱检索并返回可直接喂给 LLM 的上下文。

    - `mode=local`：实体链接 → N 跳子图 → 关系链（解决跨段落多跳问题）
    - `mode=global`：社区摘要（解决聚合统计问题）
    - `mode=auto`：按问题类型自动选择；全局问题若没命中相关社区会回退到局部
    """
    try:
        store = KnowledgeGraphStore.from_dict(get_graph_store(settings).to_dict())
        retriever = GraphRetriever(store, default_hops=request.hops, max_nodes=request.max_nodes)

        if request.mode == "local":
            result = retriever.retrieve(request.query, hops=request.hops)
        elif request.mode == "global":
            result = retriever.retrieve_global(request.query)
        else:
            result = retriever.search(request.query, hops=request.hops)

        payload = result.to_dict()
        return GraphQueryResponse(
            success=True,
            mode=payload["mode"],
            hit=payload["hit"],
            seed_count=payload["seed_count"],
            node_count=payload["node_count"],
            edge_count=payload["edge_count"],
            chain_count=payload["chain_count"],
            seeds=payload["seeds"],
            context_text=payload["context_text"],
        )
    except Exception as exc:
        logger.error("图谱检索失败: %s", exc)
        return GraphQueryResponse(success=False, error=str(exc))


@router.get("/entities/{name}", response_model=EntityDetailResponse, summary="查询实体详情与邻域")
def entity_detail(
    name: str,
    hops: int = 2,
    settings: Settings = Depends(get_settings),
):
    """返回实体本体、N 跳邻居与关联关系（含证据原文与字符区间）。"""
    try:
        store = KnowledgeGraphStore.from_dict(get_graph_store(settings).to_dict())
        entity_id = normalize_entity_name(name)
        entity = store.entities.get(entity_id)

        if entity is None:
            return EntityDetailResponse(success=True, found=False)

        neighbor_ids = store.neighbors(entity_id, hops=max(1, min(hops, 4)))
        relations = [
            rel.to_dict()
            for rel in store.relations.values()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]

        return EntityDetailResponse(
            success=True,
            found=True,
            entity=entity.to_dict(),
            neighbors=[
                {
                    "name": store.entities[nid].name,
                    "type": store.entities[nid].entity_type,
                    "degree": store.degree(nid),
                }
                for nid in sorted(neighbor_ids, key=lambda n: (-store.degree(n), n))
                if nid in store.entities
            ],
            relations=relations,
        )
    except Exception as exc:
        logger.error("查询实体详情失败: %s", exc)
        return EntityDetailResponse(success=False, error=str(exc))


@router.post("/reset", summary="清空知识图谱")
def reset_graph(
    persist: bool = True,
    settings: Settings = Depends(get_settings),
):
    """清空内存与磁盘上的图谱（重建知识库时使用）。"""
    try:
        store = get_graph_store(settings)
        store.clear()
        if persist:
            _persist(store, settings)
        from app.services.agent_qa import reset_graph_agent

        reset_graph_agent()
        return {"success": True, "message": "知识图谱已清空"}
    except Exception as exc:
        logger.error("清空图谱失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Agentic 问答
# =============================================================================

agent_router = APIRouter(prefix="/rag/agent", tags=["Agentic 问答"])


def _build_agent(settings: Settings):
    """构造 Agent（复用图谱与向量库单例）。"""
    from app.services.agent_qa import GraphRAGAgent

    # Freeze a consistent graph snapshot; admin writes cannot mutate an active answer's evidence.
    store = KnowledgeGraphStore.from_dict(get_graph_store(settings).to_dict())
    vector_store = _get_vector_store(settings)
    return GraphRAGAgent(
        graph_store=store,
        vector_store=vector_store,
        model=settings.default_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        hops=settings.graph_hops,
        top_k=5,
        max_attempts=settings.agent_max_attempts,
        max_tool_rounds=settings.agent_max_tool_rounds,
        use_llm_assessor=settings.agent_use_llm_assessor,
        thin_evidence_chars=settings.agent_thin_evidence_chars,
        grounded_threshold=settings.agent_grounded_threshold,
        price_input_per_1m=settings.llm_price_input_per_1m,
        price_output_per_1m=settings.llm_price_output_per_1m,
        llm_timeout_seconds=settings.llm_timeout_seconds,
    )


def _apply_agent_params(agent, request: AgentQARequest) -> None:
    """Apply bounded parameters to the fresh agent owned by this request."""
    agent.hops = request.hops
    agent.top_k = request.top_k
    agent.max_attempts = min(agent.max_attempts, request.max_attempts)
    agent.max_tool_rounds = min(agent.max_tool_rounds, max(1, request.max_tool_rounds))
    agent.use_llm_router = request.use_llm_router
    agent.use_llm_assessor = request.use_llm_assessor
    agent.graph_retriever.default_hops = request.hops


@agent_router.post("/qa", response_model=AgentQAResponse, summary="Agentic 问答（非流式）")
def agent_qa(
    request: AgentQARequest,
    settings: Settings = Depends(get_settings),
):
    """跑完整的 Agent 图。

    两个循环各司其职：
    - `plan → tools → assess → plan`：**补检索**。证据不足时换工具再查（解决"检索没拿到东西"）
    - `verify → rewrite`：**改问题**。引用不可接地时改写查询重试（解决"检索到了但答案编造了"）

    返回里包含 `verification`（引用可接地比例）、`tool_rounds`（实际查了几轮）、
    `usage`（LLM 调用次数 / token / 按节点归因的耗时）与 `trace`（节点轨迹），
    可直接用于评估与成本分析。
    """
    try:
        agent = _build_agent(settings)
        _apply_agent_params(agent, request)

        result = agent.answer(request.question)
        return AgentQAResponse(**result)
    except Exception as exc:
        logger.error("Agent 问答失败: %s", exc, exc_info=True)
        return AgentQAResponse(success=False, question=request.question, error=str(exc))


@agent_router.post("/qa/stream", summary="Agentic 问答（SSE 流式）")
def agent_qa_stream(
    request: AgentQARequest,
    settings: Settings = Depends(get_settings),
):
    """SSE 流式 Agent 问答。

    事件序列（与既有 `/rag/qa/stream` 向后兼容）：

        event: sources      检索来源（含 char_interval，可直接用于前端高亮）
        event: graph        图检索结果（关系链、社区摘要、命中实体）
        event: assess       充分性评估结果（decision / reason / 剩余工具）
        event: chunk        答案增量（逐 token）—— 多轮补检索时该事件会出现多次
        event: verification 引用校验结果（可接地比例、未接地项）
        event: retry        触发反思重试（附带改写后的查询）
        event: usage        LLM 用量与成本
        event: done         结束（尝试轮数、工具轮数、最终校验结果、用量）
        event: error        异常
    """
    def event_stream():
        try:
            agent = _build_agent(settings)
            _apply_agent_params(agent, request)

            for event in agent.answer_stream(request.question):
                yield (
                    f"event: {event['event']}\n"
                    f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                )
        except Exception as exc:
            logger.error("Agent 流式问答失败: %s", exc, exc_info=True)
            yield (
                "event: error\n"
                f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agent_router.get("/route-preview", summary="预览启发式检索规划（调试用）")
def route_preview(question: str):
    """只跑路由逻辑，不调用任何模型。便于调试与评测。"""
    from app.services.agent_qa import plan_route

    return {
        "question": question,
        "tools": plan_route(question),
        "is_global_query": is_global_query(question),
    }
