"""图谱检索服务 - 把知识图谱变成 LLM 能读的上下文

在 RAG 链路里的位置
------------------
    向量检索（已有）  ->  按"语义相似"召回文本块
    图谱检索（本模块）->  按"关系路径"召回跨块事实   ← GraphRAG 的增量

两者不是替代关系，而是互补：
- 向量检索擅长"某段文字里说了什么"
- 图谱检索擅长"这些实体之间是什么关系"、"整体涉及哪些对象"

本模块实现 GraphRAG 的两条检索路径：

1. **局部检索（local search）** —— `retrieve()`
   实体链接命中种子实体 -> 沿关系扩展 N 跳 -> 输出"关系链"文本。
   解决"答案分散在多个段落"的多跳问题。

2. **全局检索（global search）** —— `retrieve_global()`
   用社区发现划分子图 -> 为每个社区生成结构化摘要 -> 按与查询的相关度排序。
   解决"整体/聚合"类问题（"共涉及几家机构"），这类问题**没有任何单个文本块包含答案**，
   向量检索在原理上就不可能答对。

设计上的一个刻意选择：`context_text` 里**保留字符区间**（"字符 40-46"）。
这让 LLM 生成的答案可以被前端用已有的 `char_interval` 高亮机制锚定回原文，
不需要前端做任何改动就能复用现有的溯源能力。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.graph_store import (
    GraphEntity,
    GraphRelation,
    KnowledgeGraphStore,
    _bigrams,
)

logger = logging.getLogger(__name__)


# 全局型问题的识别模式：问"整体/聚合"而不是"某个细节"
GLOBAL_QUERY_PATTERNS = [
    r"哪些", r"有哪", r"几个", r"共.{0,3}多少", r"总共", r"一共", r"合计",
    r"整体", r"涉及", r"概况", r"总结", r"归纳", r"汇总", r"综述", r"统计",
    r"主要.*(有|是|包括)", r"都(有|是|包括)", r"全部", r"所有",
    # "多少" 后面必须跟量词才算聚合型提问
    r"多少(个|家|种|次|名|位|条|项|人|类|份|篇|台|套|张|起)",
]

# 局部型问题的模式，**优先级高于全局模式**。
#
# 为什么需要这个：中文里"多少"既用于聚合提问（"一共有多少家机构"），
# 也用于取值提问（"血压是多少"）。只靠正向模式会把后者误判成全局问题，
# 于是该走局部检索的问题跑去读社区摘要，答案自然不对。
# 这组模式显式识别"取值/方法/时间"类提问，先行排除。
LOCAL_QUERY_PATTERNS = [
    r"是(多少|什么|谁|哪一?个?|否)", r"为多少", r"等于多少",
    r"多少(钱|毫克|克|毫升|mg|ml|mmHg|度|岁|元|万|亿)",
    r"什么时候", r"多久", r"怎样", r"如何", r"为什么", r"怎么(办|做|治|回事)",
]


def is_global_query(query: str) -> bool:
    """判断是否为全局聚合型问题。

    为什么需要单独判断：全局问题的答案不在任何单个文本块里，
    必须走社区摘要而不是（或不仅是）向量检索。
    这是 GraphRAG 论文里 global search 与 local search 的分工。

    判定顺序：**先排除局部型，再匹配全局型**。
    """
    if not query:
        return False
    if any(re.search(pattern, query) for pattern in LOCAL_QUERY_PATTERNS):
        return False
    return any(re.search(pattern, query) for pattern in GLOBAL_QUERY_PATTERNS)


# =============================================================================
# 检索结果
# =============================================================================


@dataclass
class GraphRetrievalResult:
    """图谱检索结果。"""

    query: str
    mode: str = "local"  # local | global
    hops: int = 2
    seeds: List[Tuple[GraphEntity, float]] = field(default_factory=list)
    nodes: List[GraphEntity] = field(default_factory=list)
    edges: List[GraphRelation] = field(default_factory=list)
    chains: List[Dict[str, Any]] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    truncated: bool = False

    @property
    def hit(self) -> bool:
        """是否命中了图上的内容（没命中时 Agent 应回退到纯向量检索）。"""
        return bool(self.nodes or self.communities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "hops": self.hops,
            "hit": self.hit,
            "truncated": self.truncated,
            "seed_count": len(self.seeds),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "chain_count": len(self.chains),
            "community_count": len(self.communities),
            "seeds": [
                {"name": e.name, "type": e.entity_type, "link_confidence": round(score, 3)}
                for e, score in self.seeds
            ],
            "context_text": self.context_text,
        }


# =============================================================================
# 检索器
# =============================================================================


class GraphRetriever:
    """在知识图谱上做局部（多跳）与全局（社区）检索。"""

    def __init__(
        self,
        store: KnowledgeGraphStore,
        default_hops: int = 2,
        max_nodes: int = 30,
        max_chains: int = 12,
        max_communities: int = 5,
    ) -> None:
        self.store = store
        self.default_hops = default_hops
        self.max_nodes = max_nodes
        self.max_chains = max_chains
        self.max_communities = max_communities

    # -------------------------------------------------------------------------
    # 局部检索
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        hops: Optional[int] = None,
        seed_limit: int = 8,
    ) -> GraphRetrievalResult:
        """实体链接 -> 多跳扩展 -> 生成上下文。

        Args:
            query: 用户问题
            hops: 扩展跳数，None 用默认值
            seed_limit: 最多取几个种子实体

        Returns:
            GraphRetrievalResult；没命中任何实体时 `hit=False`
        """
        hops = self.default_hops if hops is None else hops
        result = GraphRetrievalResult(query=query, mode="local", hops=hops)

        if not query or not self.store.entities:
            return result

        seeds = self.store.find_entities(query, limit=seed_limit)
        if not seeds:
            logger.info("图谱检索未命中实体: %s", query)
            return result

        result.seeds = seeds
        seed_ids = [entity.entity_id for entity, _ in seeds]

        subgraph = self.store.subgraph(seed_ids, hops=hops, max_nodes=self.max_nodes)
        result.nodes = [self.store.entities[nid] for nid in subgraph["nodes"]]
        result.edges = subgraph["edges"]
        result.truncated = subgraph["truncated"]
        result.chains = self.store.relation_chains(seed_ids, max_hops=hops, max_chains=self.max_chains)
        result.context_text = self._format_local_context(result)

        logger.info(
            "图谱局部检索: 种子=%d 节点=%d 边=%d 关系链=%d",
            len(seeds), len(result.nodes), len(result.edges), len(result.chains),
        )
        return result

    def retrieve_for_entities(
        self,
        entity_names: Sequence[str],
        hops: Optional[int] = None,
    ) -> GraphRetrievalResult:
        """按显式实体名检索（供 Agent 的图检索工具直接指定实体时使用）。"""
        hops = self.default_hops if hops is None else hops
        result = GraphRetrievalResult(query=" / ".join(entity_names), mode="local", hops=hops)

        seeds: List[Tuple[GraphEntity, float]] = []
        for name in entity_names:
            entity = self.store.get_entity(name)
            if entity is not None:
                seeds.append((entity, 1.0))
        if not seeds:
            return result

        result.seeds = seeds
        seed_ids = [entity.entity_id for entity, _ in seeds]
        subgraph = self.store.subgraph(seed_ids, hops=hops, max_nodes=self.max_nodes)
        result.nodes = [self.store.entities[nid] for nid in subgraph["nodes"]]
        result.edges = subgraph["edges"]
        result.truncated = subgraph["truncated"]
        result.chains = self.store.relation_chains(seed_ids, max_hops=hops, max_chains=self.max_chains)
        result.context_text = self._format_local_context(result)
        return result

    def _format_local_context(self, result: GraphRetrievalResult) -> str:
        """把子图渲染成给 LLM 读的文本。

        格式设计考虑：
        - 关系链放最前面：它是纯向量检索拿不到的信息，应该优先进入 LLM 的注意力
        - 实体属性附上字符区间：让答案里的实体能被前端高亮回原文
        - 明确标注"共现"边的低置信度：避免 LLM 把弱证据当强事实
        """
        if not result.hit:
            return ""

        lines: List[str] = ["【知识图谱检索结果】"]

        seed_desc = "、".join(f"{e.name}({e.entity_type})" for e, _ in result.seeds)
        lines.append(f"命中的实体：{seed_desc}")

        if result.chains:
            lines.append("")
            lines.append(f"关系链（最多 {result.hops} 跳，按直接程度排序）：")
            for i, chain in enumerate(result.chains[: self.max_chains], 1):
                lines.append(f"  {i}. {chain['text']}")
        else:
            lines.append("")
            lines.append("未找到可用的关系链（图谱中该实体暂无关联）。")

        # 边的证据：带出原文片段，方便 LLM 引用与前端溯源
        strong_edges = [e for e in result.edges if e.relation_type != "共现" and e.evidence]
        if strong_edges:
            lines.append("")
            lines.append("关系证据（原文片段）：")
            for edge in strong_edges[:10]:
                source = self.store.entities.get(edge.source_id)
                target = self.store.entities.get(edge.target_id)
                if not source or not target:
                    continue
                lines.append(
                    f"  - {source.name} --[{edge.relation_type}]--> {target.name}："
                    f"「{edge.evidence[:80]}」"
                )

        weak_edges = [e for e in result.edges if e.relation_type == "共现"]
        if weak_edges:
            lines.append("")
            lines.append(
                f"（另有 {len(weak_edges)} 条「共现」弱关联，仅表示实体在同一段落出现，"
                "不能作为关系成立的依据，请勿据此下结论。）"
            )

        if result.nodes:
            lines.append("")
            lines.append("相关实体：")
            for entity in result.nodes[: self.max_nodes]:
                attrs = "；".join(f"{k}={v}" for k, v in list(entity.attributes.items())[:4])
                attr_text = f"，属性：{attrs}" if attrs else ""
                source = self._first_traceable_mention(entity)
                lines.append(
                    f"  - {entity.name}（{entity.entity_type}{attr_text}）{source}"
                )

        if result.truncated:
            lines.append("")
            lines.append("（注：子图因规模上限被截断，仅展示了关联最紧密的部分。）")

        return "\n".join(lines)

    @staticmethod
    def _first_traceable_mention(entity: GraphEntity) -> str:
        """取第一个可溯源的提及，渲染成"来源：文档 字符 x-y"。"""
        for mention in entity.mentions:
            if mention.has_position:
                title = mention.doc_title or mention.doc_id or "未知文档"
                return f"，来源：{title} 字符 {mention.start_pos}-{mention.end_pos}"
        return ""

    # -------------------------------------------------------------------------
    # 全局检索
    # -------------------------------------------------------------------------

    def retrieve_global(self, query: str = "") -> GraphRetrievalResult:
        """社区摘要式全局检索，用于回答聚合型问题。

        流程：社区发现 -> 为每个社区生成结构化摘要 -> 按与查询的相关度排序取 Top-N。

        为什么这样能回答"共涉及哪些机构"：
        向量检索会把文档切成块，任何一块都只含部分机构，模型看不到全貌；
        社区摘要把整个社区的成员实体一次性列出，聚合问题的答案就显式地出现在上下文里。
        """
        result = GraphRetrievalResult(query=query, mode="global")

        if not self.store.entities:
            return result

        communities = self.store.detect_communities()
        logger.info("全局检索: 检出 %d 个社区", len(communities))

        summaries = [self.summarize_community(comm) for comm in communities]

        # 与查询的相关度：用 bigram 交集衡量（无需 jieba，中文友好）
        if query:
            query_bigrams = _bigrams(query)
            for summary in summaries:
                summary["relevance"] = self._community_relevance(summary, query, query_bigrams)
            summaries.sort(key=lambda s: (-s["relevance"], -s["size"], s["id"]))
        else:
            summaries.sort(key=lambda s: (-s["size"], s["id"]))

        result.communities = summaries[: self.max_communities]
        result.context_text = self._format_global_context(result, total_communities=len(communities))
        return result

    def summarize_community(self, community: set) -> Dict[str, Any]:
        """为一个社区生成结构化摘要（不调用 LLM，纯确定性）。

        刻意不用 LLM 生成摘要：一是省成本、二是**可复现**——
        评测时要能跑出稳定的数字，LLM 摘要会引入随机性。
        真正需要自然语言摘要时，可以在 `agent_qa` 里对 Top-N 社区再做一次 LLM 概括。
        """
        members = [self.store.entities[eid] for eid in sorted(community) if eid in self.store.entities]

        internal_edges = [
            rel for rel in self.store.relations.values()
            if rel.source_id in community and rel.target_id in community
        ]

        # 统计实体类型分布：聚合问题的答案往往就在这里
        type_counter: Dict[str, int] = {}
        for entity in members:
            type_counter[entity.entity_type] = type_counter.get(entity.entity_type, 0) + 1

        # 社区内度数最高的几个实体 = 该社区的核心
        degree: Dict[str, int] = {}
        for rel in internal_edges:
            degree[rel.source_id] = degree.get(rel.source_id, 0) + 1
            degree[rel.target_id] = degree.get(rel.target_id, 0) + 1
        core = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

        return {
            "id": sorted(community)[0] if community else "",
            "size": len(members),
            "members": [{"name": e.name, "type": e.entity_type} for e in members],
            "entity_type_distribution": dict(
                sorted(type_counter.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "internal_relation_count": len(internal_edges),
            "core_entities": [
                {
                    "name": self.store.entities[eid].name,
                    "type": self.store.entities[eid].entity_type,
                    "degree": deg,
                }
                for eid, deg in core
                if eid in self.store.entities
            ],
            "relation_types": sorted({rel.relation_type for rel in internal_edges}),
            "relevance": 0.0,
        }

    @staticmethod
    def _community_relevance(summary: Dict[str, Any], query: str, query_bigrams: set) -> float:
        """社区与查询的相关度 = 成员实体名 / 关系类型的 bigram 与查询的重叠比例。"""
        if not query_bigrams:
            return 0.0
        tokens: set = set()
        for member in summary["members"]:
            tokens |= _bigrams(member["name"])
        for relation_type in summary["relation_types"]:
            tokens |= _bigrams(relation_type)
        if not tokens:
            return 0.0
        overlap = len(tokens & query_bigrams)
        return overlap / len(query_bigrams)

    def _format_global_context(
        self, result: GraphRetrievalResult, total_communities: int
    ) -> str:
        """把社区摘要渲染成给 LLM 读的文本。"""
        if not result.communities:
            return ""

        lines: List[str] = [
            "【知识图谱全局概况（社区摘要）】",
            f"知识图谱共划分出 {total_communities} 个社区，以下是与问题最相关的 "
            f"{len(result.communities)} 个：",
        ]

        for i, summary in enumerate(result.communities, 1):
            lines.append("")
            lines.append(
                f"社区 {i}（规模 {summary['size']}，内部关系 "
                f"{summary['internal_relation_count']} 条）："
            )
            dist = "、".join(
                f"{etype} {count} 个" for etype, count in summary["entity_type_distribution"].items()
            )
            lines.append(f"  实体类型分布：{dist}")
            if summary["core_entities"]:
                core = "、".join(f"{e['name']}({e['type']})" for e in summary["core_entities"])
                lines.append(f"  核心实体：{core}")
            if summary["relation_types"]:
                lines.append(f"  关系类型：{'、'.join(summary['relation_types'][:10])}")
            all_names = "、".join(m["name"] for m in summary["members"][:40])
            if all_names:
                lines.append(f"  全部成员实体：{all_names}")

        lines.append("")
        lines.append("（以上为全量统计，可直接用于回答数量、清单类问题。）")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 统一入口
    # -------------------------------------------------------------------------

    def search(self, query: str, hops: Optional[int] = None) -> GraphRetrievalResult:
        """按问题类型自动选择检索模式。"""
        if is_global_query(query):
            result = self.retrieve_global(query)
            # 全局模式下若社区摘要没命中相关社区，退化为局部检索兜底
            if result.hit and any(c["relevance"] > 0 for c in result.communities):
                return result
            local = self.retrieve(query, hops=hops)
            if local.hit:
                return local
            return result
        return self.retrieve(query, hops=hops)
