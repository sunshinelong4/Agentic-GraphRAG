"""检索评测：量化图检索相对切块检索的增量

用法（在 backend/ 目录下）
--------------------------
    # 纯离线，零 API 成本，秒级返回
    python -m evaluation.run_eval

    # 追加真实向量检索（需要 DASHSCOPE_API_KEY，费用极低）
    python -m evaluation.run_eval --with-vector

    # 追加 Agent 端到端评测（需要 DEEPSEEK_API_KEY）
    python -m evaluation.run_eval --live

    # 指定输出文件
    python -m evaluation.run_eval --output evaluation/results/baseline.json

指标定义
--------
`evidence_recall`（证据召回率）
    期望关键词里有多少比例出现在了**检索到的证据**中。
    这是检索阶段的端到端指标：就算 LLM 再强，证据没召回就等于零。

`single_chunk_ceiling`（单块上限）
    对同一个问题，遍历所有单块，取"单块作为证据"时的最高召回率。
    这是**任何**"切块 + 召回"方案（向量、BM25、混合重排都算）的理论上限，
    因为它们的证据永远来自有限的若干个块。

当某个问题的 `single_chunk_ceiling = 0` 时，意味着：
    **不存在任何一个块同时包含全部期望关键词 -> 切块检索原理上不可能答对。**
这类问题就是 GraphRAG 存在的理由。评测会把它们单独统计。

为什么用"关键词命中"而不是"答案文本相似度"：
    关键词命中是**确定性、可复现、零成本**的判据，不引入 LLM 评判的随机性。
    它的局限（无法判断语义等价）在 `--live` 模式下由
    `grounded_ratio`（引用可接地率）补足。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# 允许 `python evaluation/run_eval.py` 与 `python -m evaluation.run_eval` 两种跑法
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.graph_retriever import GraphRetriever, is_global_query
from app.services.graph_store import (
    KnowledgeGraphStore,
    Mention,
    RelationTriple,
    _bigrams,
    normalize_entity_name,
)
from evaluation.fixtures import CORPUS, ENTITIES, QUESTIONS, TRIPLES, corpus_by_chunk_id

logger = logging.getLogger(__name__)


# =============================================================================
# 从固定语料构建知识图谱
# =============================================================================


def build_full_documents() -> Dict[str, Dict[str, Any]]:
    """把同一个 doc_id 的块拼回"完整文档"，记录每块的偏移量。

    为什么要拼回去：真实链路里 char_interval 是相对**整篇文档**的字符下标。
    评测里如果图谱用的是"块内局部下标"，溯源校验就失真了。
    """
    documents: Dict[str, Dict[str, Any]] = {}
    for chunk in CORPUS:
        doc = documents.setdefault(
            chunk["doc_id"], {"doc_title": chunk["doc_title"], "text": "", "chunk_offsets": {}}
        )
        if doc["text"]:
            doc["text"] += "\n\n"
        doc["chunk_offsets"][chunk["chunk_id"]] = len(doc["text"])
        doc["text"] += chunk["content"]
    return documents


def build_fixture_graph() -> KnowledgeGraphStore:
    """用固定语料 + 人工三元组构建知识图谱（不调用任何 LLM）。

    刻意不调用 LLM 建图：评测数据必须**独立于被测对象**。
    如果用 LLM 抽关系再拿这张图去证明"LLM 关系抽得好"，就是循环论证。
    """
    documents = build_full_documents()
    chunks = corpus_by_chunk_id()
    store = KnowledgeGraphStore()

    # 1. 实体：名字在所属块里的位置 -> 换算成全文字符区间
    extractions_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entity in ENTITIES:
        chunk = chunks[entity["chunk_id"]]
        doc = documents[chunk["doc_id"]]
        local = chunk["content"].find(entity["name"])
        start = doc["chunk_offsets"][entity["chunk_id"]] + local if local != -1 else None
        extractions_by_doc[chunk["doc_id"]].append(
            {
                "extraction_class": entity["type"],
                "extraction_text": entity["name"],
                "attributes": entity.get("attributes") or {},
                "char_interval": (
                    {"start_pos": start, "end_pos": start + len(entity["name"])}
                    if start is not None
                    else None
                ),
            }
        )

    for doc_id, extractions in extractions_by_doc.items():
        store.add_extractions(doc_id, documents[doc_id]["doc_title"], extractions)

    # 2. 关系：证据取所属块里包含头实体的那句话，并换算成全文字符区间
    triples: List[RelationTriple] = []
    for item in TRIPLES:
        chunk = chunks[item["chunk_id"]]
        doc = documents[chunk["doc_id"]]
        evidence = _sentence_containing(chunk["content"], item["head"]) or chunk["content"][:100]
        local = chunk["content"].find(evidence)
        start = doc["chunk_offsets"][item["chunk_id"]] + local if local != -1 else None
        triples.append(
            RelationTriple(
                head=item["head"],
                relation=item["relation"],
                tail=item["tail"],
                evidence=evidence,
                confidence=0.9,
                start_pos=start,
                end_pos=start + len(evidence) if start is not None else None,
            )
        )
    # 三元组可能跨文档（如"张建国服用呋塞米"的实体来自 ck-4），
    # 统一用第一条证据所在文档作为归属
    store.add_triples(triples, doc_id="", auto_create_entities=False)

    return store


def _sentence_containing(text: str, needle: str, max_length: int = 200) -> str:
    """复用与生产代码一致的句子切分逻辑（避免评测与线上行为不一致）。"""
    from app.services.relation_extractor import _sentence_containing as impl

    return impl(text, needle, max_length)


# =============================================================================
# 关键词 BM25 基线（切块检索的代表）
#
# 为什么用字符 bigram 而不是 jieba 分词：环境装不上 jieba，而中文 bigram
# 无需词典即可用；对"切块检索"这个基线来说，分词精度不是关键变量。
# 这也让评测**零依赖、纯标准库**。
# =============================================================================


class BigramBM25:
    """基于字符 bigram 的 BM25 检索器。

    BM25 是"切块检索"的强基线：它和向量检索一样，证据只能来自有限的块，
    因此同样受 `single_chunk_ceiling` 的约束。
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = list(documents)
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()

        for doc in self.docs:
            tokens = self._tokenize(doc)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            for token in set(tokens):
                self.df[token] += 1

        self.n_docs = len(self.docs)
        self.avg_len = statistics.mean(self.doc_len) if self.doc_len else 1.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文用字符 bigram；英文/数字按词切分后再补 bigram。"""
        tokens: List[str] = []
        for word in normalize_entity_name(text).split():
            if word.isascii():
                tokens.append(word)
            tokens.extend(_bigrams(word))
        return tokens

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        # BM25 标准 idf，加 0.5 平滑避免除零与负值
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.docs:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            length = self.doc_len[idx] or 1
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self._idf(token) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:top_k]


# =============================================================================
# 指标计算
# =============================================================================


def evidence_recall(expected_terms: Sequence[str], evidence: str) -> float:
    """期望关键词在证据中的命中比例。"""
    if not expected_terms:
        return 1.0
    haystack = normalize_entity_name(evidence)
    hit = sum(1 for term in expected_terms if normalize_entity_name(term) in haystack)
    return hit / len(expected_terms)


def single_chunk_ceiling(expected_terms: Sequence[str]) -> float:
    """任何切块检索方案的理论上限。

    遍历所有单块，取最高命中比例。返回 1.0 表示"存在一个块装下了全部期望词"，
    即单块检索足够；小于 1.0 表示**必须跨块**才能凑齐证据。

    注意 `ceiling` 只约束"单块"，不约束"top_k 个块"。
    因此评测同时报告 `top_k`，并在 `--top-k` 上保持真实（默认 3 而非全取），
    这样基线不会被"一次取回大半个语料"这种不现实设置灌水。
    """
    if not CORPUS:
        return 0.0
    return max(evidence_recall(expected_terms, chunk["content"]) for chunk in CORPUS)


# =============================================================================
# 各检索配置
# =============================================================================


def make_retrievers(store: KnowledgeGraphStore, top_k: int = 5, hops: int = 2) -> Dict[str, Callable[[str], str]]:
    """构造 {配置名: 检索函数(query) -> 证据文本}。"""
    retriever = GraphRetriever(store, default_hops=hops)
    bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])

    def chunk_bm25(query: str) -> str:
        hits = bm25.search(query, top_k=top_k)
        return "\n\n".join(CORPUS[idx]["content"] for idx, _ in hits)

    def graph_local(query: str) -> str:
        return retriever.retrieve(query, hops=hops).context_text

    def graph_global(query: str) -> str:
        return retriever.retrieve_global(query).context_text

    def graph_auto(query: str) -> str:
        return retriever.search(query, hops=hops).context_text

    def hybrid(query: str) -> str:
        """切块检索 + 图检索的并集 —— 模拟真实生产里两路互补的配置。"""
        parts = [chunk_bm25(query), graph_auto(query)]
        return "\n\n".join(part for part in parts if part)

    return {
        "chunk_bm25": chunk_bm25,
        "graph_local": graph_local,
        "graph_global": graph_global,
        "graph_auto": graph_auto,
        "hybrid_bm25_graph": hybrid,
    }


def attach_vector_retriever(
    retrievers: Dict[str, Callable[[str], str]],
    top_k: int,
    hops: int,
) -> Optional[str]:
    """尝试接入真实向量检索（需要 DASHSCOPE_API_KEY）。

    **评测必须使用独立集合**：早期版本复用了生产向量库单例
    （集合名 `langextract_docs`），等于把评测语料写进用户的业务知识库，
    既污染数据、又让后续评测受历史残留影响。现在单独建 `eval_corpus` 集合。

    Returns:
        成功时返回配置名，失败时返回 None 并给出提示。
    """
    try:
        import uuid as uuid_module

        from app.config import get_settings
        from app.services.vector_store_chroma import ChromaVectorStore, DocumentChunk

        settings = get_settings()
        if not settings.dashscope_api_key:
            print("[warn] 未配置 DASHSCOPE_API_KEY，跳过向量检索配置")
            return None

        # 独立集合 + 独立持久化目录，与业务数据完全隔离
        eval_persist_dir = os.path.join(BACKEND_DIR, "evaluation", ".chroma_eval")
        store = ChromaVectorStore(
            collection_name="eval_corpus",
            persist_directory=eval_persist_dir,
            embedding_model=settings.embedding_model,
            embedding_api_key=settings.dashscope_api_key,
            embedding_base_url=settings.dashscope_base_url,
        )
        # 重建集合，保证评测起点干净（不受上次运行的残留影响）
        store.init_collection(recreate=True)

        chunks = [
            DocumentChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                doc_title=chunk["doc_title"],
                content=chunk["content"],
                chunk_type="document",
                attributes={"chunk_id": chunk["chunk_id"]},
            )
            for chunk in CORPUS
        ]
        store.add_chunks(chunks)

        def vector(query: str) -> str:
            results = store.search(query, top_k=top_k) or []
            return "\n\n".join((r.get("content") or "") for r in results)

        retrievers["vector_chroma"] = vector

        graph_retriever = GraphRetriever(build_fixture_graph(), default_hops=hops)

        def vector_plus_graph(query: str) -> str:
            parts = [vector(query), graph_retriever.search(query, hops=hops).context_text]
            return "\n\n".join(part for part in parts if part)

        retrievers["hybrid_vector_graph"] = vector_plus_graph
        return "vector_chroma"
    except Exception as exc:
        print(f"[warn] 向量检索不可用（跳过该配置）: {exc}")
        return None


# =============================================================================
# 评测主流程
# =============================================================================


def run(
    with_vector: bool = False,
    live: bool = False,
    top_k: int = 5,
    hops: int = 2,
) -> Dict[str, Any]:
    store = build_fixture_graph()
    retrievers = make_retrievers(store, top_k=top_k, hops=hops)

    if with_vector or live:
        attach_vector_retriever(retrievers, top_k=top_k, hops=hops)

    agent_metrics: Dict[str, Dict[str, Any]] = {}
    if live:
        _attach_agent(retrievers, store, metrics_sink=agent_metrics)

    graph_stats = store.stats()
    communities = store.detect_communities()
    modularity = store.modularity(communities)

    # 每个问题先算理论天花板
    ceilings = {
        item["id"]: single_chunk_ceiling(item["expected_terms"]) for item in QUESTIONS
    }
    cross_chunk = [qid for qid, ceiling in ceilings.items() if ceiling < 1.0]

    per_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for question in QUESTIONS:
        for name, retrieve in retrievers.items():
            started = time.perf_counter()
            try:
                evidence = retrieve(question["question"])
                error = ""
            except Exception as exc:
                evidence, error = "", str(exc)
            latency_ms = (time.perf_counter() - started) * 1000

            per_config[name].append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "question": question["question"],
                    "recall": evidence_recall(question["expected_terms"], evidence),
                    "ceiling": ceilings[question["id"]],
                    "latency_ms": round(latency_ms, 2),
                    "evidence_chars": len(evidence),
                    "error": error,
                }
            )

    summary = {
        name: _summarize(rows, ceilings)
        for name, rows in per_config.items()
    }

    report: Dict[str, Any] = {
        "meta": {
            "questions": len(QUESTIONS),
            "corpus_chunks": len(CORPUS),
            "graph_entities": graph_stats["entity_count"],
            "graph_relations": graph_stats["relation_count"],
            "graph_communities": len(communities),
            "graph_modularity": round(modularity, 4),
            "graph_traceable_ratio": graph_stats["traceable_ratio"],
            "cross_chunk_questions": cross_chunk,
            "top_k": top_k,
            "hops": hops,
            "live": live,
        },
        "summary": summary,
        "details": {name: rows for name, rows in per_config.items()},
    }

    # Agent 专有指标：引用可验证率（溯源质量）、反思重试、多轮工具调用与成本。
    # 这类指标无法从"证据文本"反推，必须从 Agent 的 verification/usage 字段采集。
    if live and agent_metrics:
        ratios = [m["grounded_ratio"] for m in agent_metrics.values()]
        usages = [m.get("usage") or {} for m in agent_metrics.values()]
        calls = [u.get("llm_calls", 0) for u in usages]
        tokens = [u.get("total_tokens", 0) for u in usages]
        costs = [u.get("cost") for u in usages]
        cost_configured = any(u.get("cost_configured") for u in usages)

        report["agent_metrics"] = {
            "questions_evaluated": len(agent_metrics),
            "mean_grounded_ratio": round(statistics.mean(ratios), 4) if ratios else 0.0,
            "verification_pass_rate": round(
                sum(1 for m in agent_metrics.values() if m["passed"]) / len(agent_metrics), 4
            ),
            "mean_attempts": round(
                statistics.mean([m["attempts"] for m in agent_metrics.values()]), 2
            ),
            "retried_questions": sum(1 for m in agent_metrics.values() if m["attempts"] > 1),
            "ungrounded_total": sum(len(m["ungrounded"]) for m in agent_metrics.values()),
            # ---- 多轮工具调用 ----
            "mean_tool_rounds": round(
                statistics.mean([m.get("tool_rounds", 1) for m in agent_metrics.values()]), 2
            ),
            "multi_round_questions": sum(
                1 for m in agent_metrics.values() if m.get("tool_rounds", 1) > 1
            ),
            "max_tool_rounds_used": max((m.get("tool_rounds", 1) for m in agent_metrics.values()), default=1),
            # ---- 成本 ----
            "total_llm_calls": sum(calls),
            "mean_llm_calls_per_question": round(statistics.mean(calls), 2) if calls else 0.0,
            "max_llm_calls_per_question": max(calls, default=0),
            "total_tokens": sum(tokens),
            "mean_tokens_per_question": int(statistics.mean(tokens)) if tokens else 0,
            "total_cost": round(sum(c for c in costs if c), 6) if cost_configured else None,
            "mean_cost_per_question": (
                round(statistics.mean([c for c in costs if c is not None]), 6)
                if cost_configured and any(c is not None for c in costs)
                else None
            ),
            "cost_configured": cost_configured,
            # 按节点归因：钱花在哪一步
            "tokens_by_node": _aggregate_node_usage(usages),
            "per_question": agent_metrics,
        }

    return report


def _aggregate_node_usage(usages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """把所有问题的按节点用量累加，回答"这些 token 花在哪一步"。"""
    totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
    )
    for usage in usages:
        for node, bucket in (usage.get("by_node") or {}).items():
            target = totals[node]
            target["calls"] += bucket.get("calls", 0)
            target["prompt_tokens"] += bucket.get("prompt_tokens", 0)
            target["completion_tokens"] += bucket.get("completion_tokens", 0)
            target["latency_ms"] = round(target["latency_ms"] + bucket.get("latency_ms", 0.0), 2)
    return {node: dict(bucket) for node, bucket in sorted(totals.items())}


def _summarize(rows: Sequence[Dict[str, Any]], ceilings: Dict[str, float]) -> Dict[str, Any]:
    """按问题类型分组汇总。"""
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    def mean(values: Sequence[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    result: Dict[str, Any] = {
        "overall_recall": mean([row["recall"] for row in rows]),
        "overall_latency_ms": mean([row["latency_ms"] for row in rows]),
        "by_type": {
            qtype: {
                "count": len(items),
                "recall": mean([item["recall"] for item in items]),
                "latency_ms": mean([item["latency_ms"] for item in items]),
            }
            for qtype, items in sorted(by_type.items())
        },
        "errors": sum(1 for row in rows if row["error"]),
    }

    # 单独统计"必须跨块才能凑齐证据"的问题 —— 这是最能体现实质差别的一组
    cross_rows = [row for row in rows if ceilings.get(row["id"], 1.0) < 1.0]
    result["on_cross_chunk"] = {
        "count": len(cross_rows),
        "recall": mean([row["recall"] for row in cross_rows]),
        # 有多少题真的把跨块证据**全部**召回（recall == 1.0）
        "fully_solved": sum(1 for row in cross_rows if row["recall"] >= 1.0),
    }
    return result


def _attach_agent(
    retrievers: Dict[str, Callable[[str], str]],
    store: KnowledgeGraphStore,
    metrics_sink: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """接入完整 Agent（需要 DEEPSEEK_API_KEY）。

    `metrics_sink` 用于采集**引用可验证率**等 Agent 专有指标：
    这些指标不是"召回率"的同义词，无法从返回文本里算出来，必须从 Agent 的
    `verification` 字段取，因此单独回收。
    """
    from app.services.agent_qa import GraphRAGAgent

    try:
        from app.api.rag_routes import get_vector_store
        from app.config import get_settings

        vector_store = get_vector_store(get_settings())
    except Exception:
        vector_store = None

    agent = GraphRAGAgent(graph_store=store, vector_store=vector_store)

    def agent_retrieve(query: str) -> str:
        """Agent 的最终答案是"证据的下游产物"，这里直接把答案当作证据文本。

        注意这个配置的语义与其他配置不同：其他配置测的是**检索**，
        这个配置测的是**端到端**。因此报告里要单独看，不能直接横向比 recall。
        """
        result = agent.answer(query)
        if metrics_sink is not None:
            verification = result.get("verification") or {}
            metrics_sink[query] = {
                "grounded_ratio": verification.get("grounded_ratio", 0.0),
                "passed": bool(verification.get("passed")),
                "checked_total": verification.get("checked_total", 0),
                "ungrounded": (verification.get("ungrounded_numbers") or [])
                + (verification.get("ungrounded_entities") or []),
                "attempts": result.get("attempts", 1),
                "tool_rounds": result.get("tool_rounds", 1),
                "route": result.get("route") or [],
                "usage": result.get("usage") or {},
            }
        return result.get("answer") or ""

    retrievers["agent_end_to_end"] = agent_retrieve


# =============================================================================
# 报告输出
# =============================================================================


def print_report(report: Dict[str, Any]) -> None:
    meta = report["meta"]
    print("=" * 88)
    print("GraphRAG 检索评测报告")
    print("=" * 88)
    print(
        f"问题数 {meta['questions']} | 语料块 {meta['corpus_chunks']} | "
        f"图谱实体 {meta['graph_entities']} 关系 {meta['graph_relations']}"
    )
    print(
        f"社区 {meta['graph_communities']} | 模块度 Q={meta['graph_modularity']} | "
        f"可溯源率 {meta['graph_traceable_ratio']}"
    )
    print(
        f"检索设置: top_k={meta['top_k']}（切块检索召回块数）, hops={meta['hops']}（图检索跳数）"
    )
    print(
        f"必须跨块才能凑齐证据的问题: {len(meta['cross_chunk_questions'])} 个 "
        f"({', '.join(meta['cross_chunk_questions']) or '无'})"
    )
    print()

    header = (
        f"{'配置':<22}{'总召回':>9}{'detail':>9}{'multihop':>10}{'global':>9}"
        f"{'跨块召回':>10}{'跨块全对':>10}{'延迟ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in sorted(
        report["summary"].items(), key=lambda kv: -kv[1]["overall_recall"]
    ):
        by_type = summary["by_type"]
        cross = summary["on_cross_chunk"]
        row = (
            f"{name:<22}"
            f"{summary['overall_recall']:>9.3f}"
            f"{by_type.get('detail', {}).get('recall', 0.0):>9.3f}"
            f"{by_type.get('multihop', {}).get('recall', 0.0):>10.3f}"
            f"{by_type.get('global', {}).get('recall', 0.0):>9.3f}"
            f"{cross['recall']:>10.3f}"
            f"{cross['fully_solved']:>7}/{cross['count']:<2}"
            f"{summary['overall_latency_ms']:>10.1f}"
        )
        print(row)

    print()
    print("列说明：")
    print("  总召回     = 期望关键词出现在检索证据中的平均比例")
    print("  detail     = 细节型问题（单块可答，切块检索的主场）")
    print("  multihop   = 多跳型问题（答案跨块，图检索的主场）")
    print("  global     = 聚合型问题（需汇总全量，社区摘要的主场）")
    print("  跨块召回   = 仅统计 single_chunk_ceiling<1 的问题（单块装不下全部证据）")
    print("  跨块全对   = 这些跨块问题里，期望证据被**全部**召回的数量")

    if "agent_metrics" in report:
        agent = report["agent_metrics"]
        print()
        print("-" * len(header))
        print("Agent 端到端指标（--live，需真实 API）")
        print(f"  评测问题数               : {agent['questions_evaluated']}")
        print(f"  引用可验证率（溯源质量）  : {agent['mean_grounded_ratio']:.3f}")
        print(f"  引用校验通过率           : {agent['verification_pass_rate']:.3f}")
        print(f"  反思重试：平均轮数 / 触发题数 : {agent['mean_attempts']} / {agent['retried_questions']}")
        print(f"  未被证据支撑的项总数      : {agent['ungrounded_total']}")
        print()
        print("  多轮工具调用")
        print(f"    平均工具轮数           : {agent['mean_tool_rounds']}")
        print(f"    触发多轮补检索的题数     : {agent['multi_round_questions']} / {agent['questions_evaluated']}")
        print(f"    单题最多工具轮数        : {agent['max_tool_rounds_used']}")
        print()
        print("  LLM 成本")
        print(f"    总调用次数 / 单题均值    : {agent['total_llm_calls']} / {agent['mean_llm_calls_per_question']}")
        print(f"    单题最多调用次数         : {agent['max_llm_calls_per_question']}")
        print(f"    总 token / 单题均值      : {agent['total_tokens']} / {agent['mean_tokens_per_question']}")
        if agent["cost_configured"]:
            print(f"    总费用 / 单题均值        : {agent['total_cost']} / {agent['mean_cost_per_question']}")
        else:
            print("    费用                    : 未配置单价（设 LLM_PRICE_INPUT_PER_1M / "
                  "LLM_PRICE_OUTPUT_PER_1M 后自动计算）")
        print("    token 按节点归因：")
        for node, bucket in (agent.get("tokens_by_node") or {}).items():
            print(
                f"      {node:<14} 调用 {bucket['calls']:>3} 次  "
                f"输入 {bucket['prompt_tokens']:>7}  输出 {bucket['completion_tokens']:>6}  "
                f"耗时 {bucket['latency_ms']:>9.0f}ms"
            )
        print()
        print("  说明：可验证率 = 答案里的数字与实体名能在检索证据中找到的比例，")
        print("        它衡量的是「答案有没有编造」，与召回率是两件事，不能互相替代。")
        print("        多轮成本是这个特性的真实代价 —— 报告出来是为了让权衡可见。")


def main() -> int:
    parser = argparse.ArgumentParser(description="GraphRAG 检索评测")
    parser.add_argument("--with-vector", action="store_true", help="接入真实向量检索（需 DASHSCOPE_API_KEY）")
    parser.add_argument("--live", action="store_true", help="接入完整 Agent（需 DEEPSEEK_API_KEY）")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="切块检索召回块数。默认 3 而非 5：语料有 22 个块，取 5 会让基线"
             "被'一次取回近四分之一语料'这种不现实的设置灌水，掩盖跨块差距。",
    )
    parser.add_argument("--hops", type=int, default=2, help="图检索跳数")
    parser.add_argument("--output", type=str, default="", help="结果 JSON 输出路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    report = run(with_vector=args.with_vector, live=args.live, top_k=args.top_k, hops=args.hops)
    print_report(report)

    # 按模式分文件保存：否则在线评测的结果会被下一次离线评测覆盖，
    # 而"离线可复现"与"真实 API 实测"是两份需要同时留存的证据。
    if args.output:
        output = args.output
    else:
        suffix = "offline"
        if args.live:
            suffix = "live"
        elif args.with_vector:
            suffix = "vector"
        output = os.path.join(BACKEND_DIR, "evaluation", "results", f"{suffix}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
