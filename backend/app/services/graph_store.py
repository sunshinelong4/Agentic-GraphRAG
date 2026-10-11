"""知识图谱存储服务 - 纯标准库实现（零外部图数据库依赖）

设计目标
--------
1. **后端真正持有知识图谱**。改造前，"图谱"只存在于前端 `KnowledgeGraph.tsx`，
   由 `generateRelationshipsFromItems()` 里的硬编码规则（`人物 + 机构 -> "属于"`）
   临时拼出来，既不持久化、也不参与检索。本模块把图谱落到后端。
2. **支持多跳子图检索**（GraphRAG 的 local search）：从一个查询命中的实体出发，
   沿关系扩展 N 跳，把"关系链"作为额外上下文交给 LLM。
3. **支持社区发现**（GraphRAG 的 global search 基础）：用标签传播算法划分子图，
   可回答"整体上涉及哪些机构/主题"这类聚合型问题。
4. **零外部依赖**：不引入 Neo4j / networkx，只用 dict + BFS + difflib。
   好处是可读、可测、部署简单；代价是超大规模（百万级边）需要换实现。

数据模型
--------
    Mention        -- 实体在原文中的一次出现（doc_id + 字符区间），溯源的最小单位
    GraphEntity    -- 一个规范化后的实体；不同写法会归并到同一实体（别名归并）
    GraphRelation  -- 一条有向关系 source -> target，携带原文证据与置信度

与既有代码的关系
----------------
- 上游：`app/core/extractor.py` 产出的 extractions（含 `char_interval` 溯源信息）
- 下游：`app/services/graph_retriever.py` 做多跳检索
        `app/services/agent_qa.py` 把图检索作为 Agent 的一个工具
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# 实体名规范化
# =============================================================================

# 中文书名号 / 引号 / 括号等包裹符号，规范化时剥离
_WRAPPER_CHARS = "《》「」『』“”‘’\"'（）()【】[]〈〉"

# 归一化后长度低于该值的实体不进入图谱（噪声太大，如单字"的"）
MIN_ENTITY_LENGTH = 2


def normalize_entity_name(name: str) -> str:
    """把实体名规范化成稳定的 entity_id。

    规则（顺序敏感）：
    1. 全角转半角（NFKC）—— "ＡＢＣ" 与 "ABC" 归并
    2. 剥离书名号/引号/括号等包裹符号 —— "《王者荣耀》" 与 "王者荣耀" 归并
    3. 折叠空白（含全角空格）
    4. ASCII 部分转小写 —— "IBM" 与 "ibm" 归并；中文不受影响

    Args:
        name: 原始实体名

    Returns:
        规范化后的字符串。真实生产系统通常用 UUID 作主键 + 别名表，
        这里直接用规范化名作主键，换来的是**可读、可调试**——
        打开 JSON 就能看懂图谱里有什么。
    """
    if not name:
        return ""
    # 1. 全角 -> 半角
    text = unicodedata.normalize("NFKC", str(name))
    # 2. 剥离包裹符号（逐个字符处理，保留内部字符）
    text = "".join(ch for ch in text if ch not in _WRAPPER_CHARS)
    # 3. 折叠空白
    text = re.sub(r"\s+", " ", text).strip()
    # 4. ASCII 小写（中文原样保留）
    return _casefold_ascii_only(text)


def _casefold_ascii_only(text: str) -> str:
    """只对 ASCII 字符做小写转换，中文原样保留。"""
    return "".join(ch.lower() if ch.isascii() else ch for ch in text)


def is_valid_entity_name(name: str) -> bool:
    """判断一个规范化后的名字是否值得进入图谱。"""
    if not name:
        return False
    if len(name) < MIN_ENTITY_LENGTH:
        return False
    # 纯符号/纯空白不进图
    if not any(ch.isalnum() for ch in name):
        return False
    return True


# =============================================================================
# 轻量文本相似度工具
#
# 为什么不用 jieba / rank_bm25：这个环境装不上新依赖，而且中文用**字符 bigram**
# 做近似切分效果已经够用（无需词典）。这两个函数后续做混合检索时还会复用。
# =============================================================================


def _bigrams(text: str) -> Set[str]:
    """字符二元组集合。中文里充当"无需词典的分词"。

    "扩张型心肌病" -> {"扩张", "张型", "型心", "心肌", "肌病"}
    单字文本退化为单字集合，避免空集导致永远无法匹配。
    """
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _best_window_ratio(needle: str, haystack: str) -> float:
    """在 haystack 上滑动与 needle 等长（±1）的窗口，返回最高相似度。

    直接 `SequenceMatcher(needle, haystack)` 在 haystack 远长于 needle 时
    会把相似度稀释掉：'扩张型心肌病' vs '扩张性心肌病怎么治' 只有 0.625，
    而滑窗能拿到 0.833，正好跨过 0.72 的阈值。
    """
    n = len(needle)
    best = 0.0
    for width in {max(2, n - 1), n, n + 1}:
        if width > len(haystack):
            continue
        for start in range(len(haystack) - width + 1):
            ratio = SequenceMatcher(None, needle, haystack[start:start + width]).ratio()
            if ratio > best:
                best = ratio
                if best == 1.0:
                    return best
    return best


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class Mention:
    """实体在原文中的一次出现（溯源锚点）。"""

    doc_id: str
    doc_title: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    surface: str = ""  # 原文里的原始写法（可能和规范化名不同）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_position(self) -> bool:
        """是否带字符区间。GraphRAG 里"可溯源"的判定依据。"""
        return self.start_pos is not None and self.end_pos is not None


@dataclass
class GraphEntity:
    """图谱中的一个实体节点。"""

    entity_id: str
    name: str  # 展示名（首次出现的写法）
    entity_type: str = "未知"
    attributes: Dict[str, Any] = field(default_factory=dict)
    mentions: List[Mention] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)  # 见过的其它写法

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["aliases"] = sorted(self.aliases)
        return data

    @property
    def mention_count(self) -> int:
        return len(self.mentions)

    @property
    def source_docs(self) -> List[str]:
        seen: List[str] = []
        for m in self.mentions:
            if m.doc_id not in seen:
                seen.append(m.doc_id)
        return seen

    @property
    def traceable_ratio(self) -> float:
        """提及中带字符区间的比例 —— 衡量溯源质量的核心指标。"""
        if not self.mentions:
            return 0.0
        return sum(1 for m in self.mentions if m.has_position) / len(self.mentions)


@dataclass
class GraphRelation:
    """图谱中的一条有向关系边。"""

    source_id: str
    target_id: str
    relation_type: str
    evidence: str = ""  # 支撑该关系的原文片段
    doc_id: str = ""
    confidence: float = 0.8
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> Tuple[str, str, str]:
        """去重键：同源、同目标、同关系类型视为同一条边。"""
        return (self.source_id, self.target_id, self.relation_type)


@dataclass
class RelationTriple:
    """关系抽取器的输出（尚未落图）。

    `start_pos` / `end_pos` 指向**证据片段在原文中的字符区间**。
    有了它，关系也能像实体一样被溯源与前端高亮 —— 这是"每条结论都能跳回原文"
    这一产品承诺的另一半（实体那一半由 `Mention` 承担）。
    """

    head: str
    relation: str
    tail: str
    head_type: str = ""
    tail_type: str = ""
    evidence: str = ""
    confidence: float = 0.8
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


# =============================================================================
# 图谱存储
# =============================================================================


class KnowledgeGraphStore:
    """内存知识图谱 + JSON 持久化。

    线程安全：所有写操作加 `RLock`。FastAPI 的同步路由跑在线程池里，
    没有锁的话 `add_extractions` 与 `stats` 并发会读到半更新的 dict。
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: Dict[Tuple[str, str, str], GraphRelation] = {}
        # 邻接表（无向视图，用于多跳扩展与社区发现）
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        # 有向边类型索引: (source, target) -> relation_type
        # 为什么需要：DFS 枚举关系链时会高频查询"这两点之间是什么关系"，
        # 直接遍历 self.relations 是 O(E)，会让多跳检索慢到不可用。
        self._edge_types: Dict[Tuple[str, str], str] = {}
        self.persist_path = persist_path
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # 基础读写
    # -------------------------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "未知",
        attributes: Optional[Dict[str, Any]] = None,
        mention: Optional[Mention] = None,
    ) -> Optional[str]:
        """写入/更新一个实体，返回 entity_id；名字无效时返回 None。"""
        entity_id = normalize_entity_name(name)
        if not is_valid_entity_name(entity_id):
            return None

        with self._lock:
            entity = self.entities.get(entity_id)
            if entity is None:
                entity = GraphEntity(
                    entity_id=entity_id,
                    name=str(name).strip(),
                    entity_type=entity_type or "未知",
                    attributes=dict(attributes or {}),
                )
                self.entities[entity_id] = entity
                self._adj.setdefault(entity_id, set())
            else:
                # 类型冲突时保留先到的具体类型，避免"未知"覆盖"药物"
                if entity.entity_type in ("", "未知") and entity_type:
                    entity.entity_type = entity_type
                if attributes:
                    for k, v in attributes.items():
                        entity.attributes.setdefault(k, v)

            # 记录别名（原文写法）
            if mention and mention.surface:
                entity.aliases.add(mention.surface)
            if mention:
                entity.mentions.append(mention)

            return entity_id

    def upsert_relation(self, relation: GraphRelation) -> bool:
        """写入一条关系。自环、悬空端点会被拒绝；重复边会合并证据并取较高置信度。"""
        if not relation.source_id or not relation.target_id:
            return False
        if relation.source_id == relation.target_id:
            return False  # 自环对检索无价值，且会污染社区发现
        if relation.source_id not in self.entities or relation.target_id not in self.entities:
            return False  # 悬空端点说明关系抽取幻觉了，直接丢弃

        with self._lock:
            existing = self.relations.get(relation.key)
            if existing is None:
                self.relations[relation.key] = relation
            else:
                # 同一条边被多次抽到 -> 提升置信度，但封顶
                existing.confidence = min(1.0, max(existing.confidence, relation.confidence) + 0.05)
                if relation.evidence and relation.evidence not in existing.evidence:
                    existing.evidence = (existing.evidence + " / " + relation.evidence).strip(" /")
            self._adj[relation.source_id].add(relation.target_id)
            self._adj[relation.target_id].add(relation.source_id)
            self._edge_types.setdefault(
                (relation.source_id, relation.target_id), relation.relation_type
            )
            return True

    # -------------------------------------------------------------------------
    # 入口 1：从 LangExtract 抽取结果建图
    # -------------------------------------------------------------------------

    def add_extractions(
        self,
        doc_id: str,
        doc_title: str,
        extractions: Sequence[Dict[str, Any]],
        entity_classes: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """把抽取结果中的实体写入图谱。

        Args:
            doc_id: 文档 ID
            doc_title: 文档标题
            extractions: `Extractor.extract()` 返回的 extractions 列表，
                每项含 extraction_class / extraction_text / attributes / char_interval
            entity_classes: 只把这些类别当作实体；None 表示全部当作实体

        Returns:
            统计信息 {"entities_added": int, "entities_merged": int, "skipped": int}
        """
        added = merged = skipped = 0

        with self._lock:
            for ext in extractions:
                cls = ext.get("extraction_class") or "未知"
                text = (ext.get("extraction_text") or "").strip()
                if not text:
                    skipped += 1
                    continue
                if entity_classes is not None and cls not in entity_classes:
                    skipped += 1
                    continue

                interval = ext.get("char_interval") or {}
                start = interval.get("start_pos")
                end = interval.get("end_pos")
                # 有些抽取结果没有位置信息，此时只丢了"可溯源"这一半能力
                if start is None or end is None:
                    start = end = None

                entity_id = normalize_entity_name(text)
                existed = entity_id in self.entities

                self.upsert_entity(
                    name=text,
                    entity_type=cls,
                    attributes=ext.get("attributes") or {},
                    mention=Mention(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        start_pos=start,
                        end_pos=end,
                        surface=text,
                    ),
                )
                if not is_valid_entity_name(entity_id):
                    skipped += 1
                elif existed:
                    merged += 1
                else:
                    added += 1

        result = {"entities_added": added, "entities_merged": merged, "skipped": skipped}
        logger.info(
            "图谱写入抽取结果: doc=%s 新增实体=%d 归并=%d 跳过=%d",
            doc_title, added, merged, skipped,
        )
        return result

    # -------------------------------------------------------------------------
    # 入口 2：写入关系三元组
    # -------------------------------------------------------------------------

    def add_triples(
        self,
        triples: Sequence[RelationTriple],
        doc_id: str = "",
        doc_title: str = "",
        auto_create_entities: bool = True,
    ) -> Dict[str, Any]:
        """把关系三元组写入图谱。

        Args:
            triples: 关系三元组列表
            doc_id: 来源文档 ID
            doc_title: 来源文档标题
            auto_create_entities: 端点实体不存在时是否自动创建。
                注意：自动创建会把 LLM 幻觉出来的实体也建进图里，
                所以 `relation_extractor` 已在上游做过一轮"端点必须在已知实体集合内"的校验。
                真实链路里应当先 `add_extractions` 建实体（带字符区间、可溯源），
                再用 `auto_create_entities=False` 只补边。

        Returns:
            {"relations_added": int, "relations_merged": int, "rejected": int}
        """
        added = merged = rejected = 0

        with self._lock:
            for triple in triples:
                if auto_create_entities:
                    # 补一条无位置的 mention：虽然不可溯源（会拉低 traceable_ratio），
                    # 但能让 document_count 统计正确 —— 否则"只有关系没有实体"的
                    # 文档在图谱里会消失。这个坑是测试里 document_count 断言 0==1 暴露的。
                    for name, etype in ((triple.head, triple.head_type), (triple.tail, triple.tail_type)):
                        self.upsert_entity(
                            name,
                            etype or "未知",
                            mention=Mention(
                                doc_id=doc_id, doc_title=doc_title, surface=str(name).strip()
                            ) if doc_id else None,
                        )

                source_id = normalize_entity_name(triple.head)
                target_id = normalize_entity_name(triple.tail)

                relation = GraphRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=triple.relation or "相关",
                    evidence=triple.evidence,
                    doc_id=doc_id,
                    confidence=triple.confidence,
                    start_pos=triple.start_pos,
                    end_pos=triple.end_pos,
                )
                duplicated = relation.key in self.relations
                if self.upsert_relation(relation):
                    if duplicated:
                        merged += 1
                    else:
                        added += 1
                else:
                    rejected += 1

        result = {"relations_added": added, "relations_merged": merged, "rejected": rejected}
        logger.info("图谱写入关系: 新增=%d 合并=%d 拒绝=%d", added, merged, rejected)
        return result

    # -------------------------------------------------------------------------
    # 实体链接：把自然语言查询映射到图上的实体
    # -------------------------------------------------------------------------

    def find_entities(
        self,
        query: str,
        limit: int = 10,
        fuzzy_threshold: float = 0.72,
    ) -> List[Tuple[GraphEntity, float]]:
        """从查询文本里找出被提到的实体（实体链接 / entity linking）。

        三级匹配，按置信度降序：
        1. 精确包含：实体名是查询的子串 —— 置信度 1.0（长实体优先）
        2. 规范化后包含 —— 置信度 0.9
        3. 模糊匹配（difflib 相似度 >= 阈值）—— 置信度 = 相似度 * 0.8
           用于错别字与简称场景，例如查询"扩张性心肌病"应命中"扩张型心肌病"

        第 3 步的两个工程要点（都是踩过坑才补上的）：
        - **必须在查询上滑窗**，而不是拿整个查询跟实体名比相似度。
          查询通常比实体名长得多，直接比会把相似度稀释到阈值以下，
          导致模糊匹配形同虚设。
        - **先用字符 bigram 做预筛**。滑窗比较是 O(len(query)) 次
          SequenceMatcher，entities × 窗口数会轻松上到几十万次调用。
          先用 bigram 交集挡掉绝大部分不可能匹配的实体。

        Returns:
            [(实体, 置信度), ...]，按置信度降序、名称长度降序
        """
        if not query or not self.entities:
            return []

        hits: Dict[str, float] = {}
        lowered = query.lower()
        query_bigrams = _bigrams(lowered)

        with self._lock:
            for entity_id, entity in self.entities.items():
                # 1/2. 包含匹配：优先用展示名和别名
                candidates = [entity.name.lower()] + [a.lower() for a in entity.aliases]
                best = 0.0
                for cand in candidates:
                    if not cand:
                        continue
                    if cand in lowered:
                        # 越长的实体名命中越可信
                        best = max(best, 1.0 if len(cand) >= 3 else 0.85)
                        break
                if best == 0.0 and entity_id in lowered:
                    best = 0.9
                if best > 0.0:
                    hits[entity_id] = best
                    continue

                # 3. 模糊匹配：先 bigram 预筛，再做滑窗相似度
                if len(entity_id) >= 3 and _bigrams(entity_id) & query_bigrams:
                    ratio = _best_window_ratio(entity_id, lowered)
                    if ratio >= fuzzy_threshold:
                        hits[entity_id] = ratio * 0.8

        scored = [(self.entities[eid], score) for eid, score in hits.items()]
        # 置信度相同则长实体优先（更具体），再按名字排序保证确定性
        scored.sort(key=lambda pair: (-pair[1], -len(pair[0].entity_id), pair[0].entity_id))
        return scored[:limit]

    def get_entity(self, name_or_id: str) -> Optional[GraphEntity]:
        """按名字或 id 取实体。"""
        return self.entities.get(normalize_entity_name(name_or_id))

    def degree(self, entity_id: str) -> int:
        """实体的度数（关联的实体数量，无向视图）。

        对外暴露为公开方法：图检索、API 层排序都要用它，
        外部直接读 `_adj` 会绕过封装。
        """
        return len(self._adj.get(entity_id, ()))

    # -------------------------------------------------------------------------
    # 多跳检索：GraphRAG 的 local search 核心
    # -------------------------------------------------------------------------

    def neighbors(self, entity_id: str, hops: int = 1) -> Set[str]:
        """返回 N 跳内的所有实体 id（不含起点）。"""
        if entity_id not in self._adj:
            return set()
        visited = {entity_id}
        frontier = deque([(entity_id, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for nb in self._adj.get(node, ()):  # 无向视图：双向扩展
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, depth + 1))
        visited.discard(entity_id)
        return visited

    def subgraph(
        self,
        seed_ids: Iterable[str],
        hops: int = 2,
        max_nodes: int = 30,
    ) -> Dict[str, Any]:
        """从种子实体出发扩展出一个子图（BFS + 节点预算）。

        为什么要预算：一个热门实体 2 跳可能带出几百个节点，全塞给 LLM 会爆上下文、
        也会稀释相关性。这里按"跳数 → 度数"排序，优先保留图上更核心的实体。

        Args:
            seed_ids: 种子实体 id 集合
            hops: 扩展跳数
            max_nodes: 子图节点上限

        Returns:
            {
              "nodes": [entity_id, ...],
              "edges": [GraphRelation, ...],
              "seed_ids": [...],
              "truncated": bool,
            }
        """
        seeds = [s for s in dict.fromkeys(seed_ids) if s in self.entities]
        if not seeds:
            return {"nodes": [], "edges": [], "seed_ids": [], "truncated": False}

        with self._lock:
            # 按跳数分层 BFS，同一层内按度数降序（度数高 = 图上的枢纽实体）
            ordered: List[str] = list(seeds)
            seen: Set[str] = set(seeds)
            frontier: List[str] = list(seeds)
            truncated = False

            for _ in range(max(hops, 0)):
                next_layer: List[str] = []
                for node in frontier:
                    for nb in self._adj.get(node, ()):
                        if nb not in seen:
                            seen.add(nb)
                            next_layer.append(nb)
                if not next_layer:
                    break
                next_layer.sort(key=lambda nid: (-len(self._adj.get(nid, ())), nid))
                remaining = max_nodes - len(ordered)
                if remaining <= 0:
                    truncated = True
                    break
                if len(next_layer) > remaining:
                    next_layer = next_layer[:remaining]
                    truncated = True
                ordered.extend(next_layer)
                frontier = next_layer

            node_set = set(ordered)
            edges = [
                rel for rel in self.relations.values()
                if rel.source_id in node_set and rel.target_id in node_set
            ]

        return {
            "nodes": ordered,
            "edges": edges,
            "seed_ids": seeds,
            "truncated": truncated,
        }

    def relation_chains(
        self,
        seed_ids: Iterable[str],
        max_hops: int = 2,
        max_chains: int = 12,
    ) -> List[Dict[str, Any]]:
        """枚举从种子实体出发的关系链（可读的多跳路径）。

        这是把"图"变成"LLM 能读的上下文"的关键一步。输出形如：

            患者:张建国 --[患有]--> 扩张型心肌病 --[治疗]--> 沙库巴曲缬沙坦钠片

        纯向量检索永远无法回答"张三和李四之间是什么关系"这类问题，
        因为它检索的是**相似文本**，而这类问题的答案在**关系路径**上。

        Returns:
            [{"chain": [{"entity": name, "type": ...}, ...],
              "relations": [relation_type, ...],
              "text": "A --[r1]--> B --[r2]--> C"}, ...]
        """
        seeds = [s for s in dict.fromkeys(seed_ids) if s in self.entities]
        if not seeds:
            return []

        results: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[str, ...]] = set()

        with self._lock:
            # DFS 枚举简单路径（不重复访问节点），避免环形图爆炸
            def walk(path: List[str], rels: List[str], depth: int) -> None:
                if len(results) >= max_chains or depth > max_hops:
                    return
                if depth >= 1:
                    signature = tuple(path)
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        results.append(self._format_chain(path, rels))
                        if len(results) >= max_chains:
                            return

                if depth == max_hops:
                    return

                last = path[-1]
                # 邻居按度数降序 + 名字排序 → 确定性输出，且优先走枢纽
                candidates = sorted(
                    self._adj.get(last, ()),
                    key=lambda nid: (-len(self._adj.get(nid, ())), nid),
                )
                for nb in candidates:
                    if nb in path:
                        continue
                    rel_type = self._relation_type_between(last, nb)
                    walk(path + [nb], rels + [rel_type], depth + 1)
                    if len(results) >= max_chains:
                        return

            for seed in seeds:
                walk([seed], [], 0)
                if len(results) >= max_chains:
                    break

        # 短链优先（更直接、噪声更少）
        results.sort(key=lambda item: (len(item["chain"]), item["text"]))
        return results[:max_chains]

    def _relation_type_between(self, a: str, b: str) -> str:
        """取 a、b 之间任意一条边的类型（有向优先，再取反向）。O(1) 查索引。"""
        forward = self._edge_types.get((a, b))
        if forward:
            return forward
        backward = self._edge_types.get((b, a))
        if backward:
            return backward
        return "相关"

    def _format_chain(self, path: List[str], rels: List[str]) -> Dict[str, Any]:
        """把路径格式化成可读文本。"""
        chain = []
        for nid in path:
            entity = self.entities[nid]
            chain.append({"entity_id": nid, "name": entity.name, "type": entity.entity_type})

        parts = [f"{chain[0]['name']}({chain[0]['type']})"]
        for i, rel in enumerate(rels):
            parts.append(f"--[{rel}]-->")
            parts.append(f"{chain[i + 1]['name']}({chain[i + 1]['type']})")

        return {
            "chain": chain,
            "relations": rels,
            "hops": len(rels),
            "text": " ".join(parts),
        }

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> Optional[Dict[str, Any]]:
        """两个实体之间的最短关系路径（BFS）。用于"X 和 Y 什么关系"这类问题。"""
        src = normalize_entity_name(source)
        dst = normalize_entity_name(target)
        if src not in self.entities or dst not in self.entities:
            return None

        with self._lock:
            parent: Dict[str, Optional[str]] = {src: None}
            frontier = deque([(src, 0)])
            while frontier:
                node, depth = frontier.popleft()
                if node == dst:
                    path = []
                    cur: Optional[str] = node
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    rels = [
                        self._relation_type_between(path[i], path[i + 1])
                        for i in range(len(path) - 1)
                    ]
                    return self._format_chain(path, rels)
                if depth >= max_hops:
                    continue
                for nb in sorted(self._adj.get(node, ())):
                    if nb not in parent:
                        parent[nb] = node
                        frontier.append((nb, depth + 1))
        return None

    # -------------------------------------------------------------------------
    # 社区发现：GraphRAG 的 global search 基础
    # -------------------------------------------------------------------------

    def detect_communities(self, max_iter: int = 50) -> List[Set[str]]:
        """社区发现：Louvain 第一层（局部移动 + 模块度优化）。

        为什么需要社区：
        向量检索擅长"局部问题"（某个实体的细节），但对"这份文档整体涉及哪些机构"
        这类**全局聚合问题**无能为力——因为没有任何单个 chunk 包含答案。
        GraphRAG 的做法是先把图切成社区，再为每个社区生成摘要，
        聚合问题就从"检索文本"变成"读摘要"。

        算法选择（这里有个真实的踩坑记录）：
        最初我用**标签传播（LPA）**，但在"两个三角簇 + 一条桥边"这种图上，
        LPA 会一路把标签传播过去、最终塌缩成一个巨型社区，模块度算出正好 0.0。
        这是 LPA 的已知缺陷。改用**模块度优化**（Louvain 的局部移动阶段）后，
        同样这张图得到 Q≈0.36，两个簇被正确切开。

        局部移动的增益公式（无向图，边权为 1）：
            把孤立节点 i 并入社区 C 的模块度增益
            ΔQ = k_i,in / m  -  Σ_tot * k_i / (2 m²)
          其中 k_i,in = i 到 C 内部的边数，Σ_tot = C 内所有节点度数之和，m = 总边数。
        每个节点依次尝试"留在原社区 / 并入邻居社区 / 自己单飞"，取增益最大者，
        直到一轮下来没有任何节点移动。

        未做的部分（诚实标注）：完整 Louvain 还有"图聚合"第二层，
        把每个社区收缩成一个超级节点后递归。本实现在中小规模图上
        效果已接近完整 Louvain；超大图应改用 Leiden（networkx / graspologic）。

        Returns:
            社区列表（元素为 entity_id 集合），按规模降序
        """
        with self._lock:
            nodes = sorted(self._adj)
            if not nodes:
                return []

            degree = {n: len(self._adj[n]) for n in nodes}
            m = len(self.relations)
            if m == 0:
                # 没有边 -> 每个节点自成一个社区
                return [{n} for n in nodes]

            community: Dict[str, str] = {n: n for n in nodes}
            total_degree: Dict[str, int] = {n: degree[n] for n in nodes}  # Σ_tot per community

            for _ in range(max_iter):
                moved = False
                for node in nodes:
                    if degree[node] == 0:
                        continue

                    # 统计到各邻居社区的边数
                    weight_to_comm: Dict[str, int] = defaultdict(int)
                    for nb in self._adj[node]:
                        weight_to_comm[community[nb]] += 1

                    current = community[node]
                    # 先把 node 从原社区摘出来，再评估各去处
                    total_degree[current] -= degree[node]

                    w_current = weight_to_comm.get(current, 0)
                    best_comm = current
                    best_gain = (
                        w_current / m
                        - (total_degree[current] * degree[node]) / (2 * m * m)
                    )

                    for comm, weight in weight_to_comm.items():
                        if comm == current:
                            continue
                        gain = weight / m - (total_degree[comm] * degree[node]) / (2 * m * m)
                        # 增益更优则采纳；增益相同取字典序更小的社区 id，保证可复现
                        if gain > best_gain + 1e-12:
                            best_gain, best_comm = gain, comm
                        elif abs(gain - best_gain) <= 1e-12 and comm < best_comm:
                            best_comm = comm

                    if best_gain <= 0:
                        # 并入任何社区都不划算 -> 自己单飞
                        best_comm = node
                        total_degree.setdefault(node, 0)

                    community[node] = best_comm
                    total_degree[best_comm] = total_degree.get(best_comm, 0) + degree[node]
                    if best_comm != current:
                        moved = True

                if not moved:
                    break

            # 按社区归组，并把社区 id 规范化为"内部字典序最小的成员"，保证输出稳定
            groups: Dict[str, Set[str]] = defaultdict(set)
            for node, comm in community.items():
                groups[comm].add(node)

        return sorted(groups.values(), key=lambda s: (-len(s), sorted(s)[0]))

    def modularity(self, communities: Sequence[Set[str]]) -> float:
        """计算划分的模块度 Q（衡量社区结构显著性）。

        Q = Σ_c [ L_c/m - (k_c/(2m))² ]
          L_c: 社区 c 内部的边数
          k_c: 社区 c 内所有节点度数之和
          m  : 全图边数

        Q > 0.3 通常认为存在有意义的社区结构。这个数字可以**直接写进简历**
        （"社区模块度 Q=0.xx"），因为它客观、可复现。
        """
        m = len(self.relations)
        if m == 0:
            return 0.0

        node_to_comm: Dict[str, int] = {}
        for idx, comm in enumerate(communities):
            for nid in comm:
                node_to_comm[nid] = idx

        internal_edges = [0] * len(communities)
        degree_sum = [0] * len(communities)

        for rel in self.relations.values():
            ci = node_to_comm.get(rel.source_id)
            cj = node_to_comm.get(rel.target_id)
            if ci is not None:
                degree_sum[ci] += 1
            if cj is not None:
                degree_sum[cj] += 1
            if ci is not None and ci == cj:
                internal_edges[ci] += 1

        q = 0.0
        for idx in range(len(communities)):
            if degree_sum[idx] == 0:
                continue
            q += (internal_edges[idx] / m) - (degree_sum[idx] / (2 * m)) ** 2
        return q

    # -------------------------------------------------------------------------
    # 统计与持久化
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """图谱统计信息。既是运维指标，也是简历上的数字来源。"""
        with self._lock:
            total_mentions = sum(e.mention_count for e in self.entities.values())
            traceable = sum(
                1 for e in self.entities.values() for m in e.mentions if m.has_position
            )
            type_counts = Counter(e.entity_type for e in self.entities.values())
            doc_ids = {m.doc_id for e in self.entities.values() for m in e.mentions}
            return {
                "entity_count": len(self.entities),
                "relation_count": len(self.relations),
                "document_count": len(doc_ids),
                "mention_count": total_mentions,
                "traceable_mention_count": traceable,
                "traceable_ratio": round(traceable / total_mentions, 4) if total_mentions else 0.0,
                "entity_type_distribution": dict(type_counts.most_common()),
                "avg_degree": round(
                    2 * len(self.relations) / len(self.entities), 3
                ) if self.entities else 0.0,
            }

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
                "relations": [rel.to_dict() for rel in self.relations.values()],
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraphStore":
        store = cls()
        for entity_id, raw in (data.get("entities") or {}).items():
            mentions = [Mention(**m) for m in raw.get("mentions", [])]
            store.entities[entity_id] = GraphEntity(
                entity_id=entity_id,
                name=raw.get("name", entity_id),
                entity_type=raw.get("entity_type", "未知"),
                attributes=raw.get("attributes") or {},
                mentions=mentions,
                aliases=set(raw.get("aliases") or []),
            )
            store._adj.setdefault(entity_id, set())

        for raw in data.get("relations") or []:
            rel = GraphRelation(**raw)
            store.relations[rel.key] = rel
            store._adj[rel.source_id].add(rel.target_id)
            store._adj[rel.target_id].add(rel.source_id)
            store._edge_types.setdefault((rel.source_id, rel.target_id), rel.relation_type)
        return store

    def save(self, path: Optional[str] = None) -> str:
        """持久化为 JSON。"""
        target = path or self.persist_path
        if not target:
            raise ValueError("未指定图谱持久化路径")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        # Serialize snapshots and replace atomically: readers never see half-written JSON.
        with self._lock:
            payload = self.to_dict()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(os.path.abspath(target)), delete=False) as f:
                    temporary = f.name
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, target)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        logger.info("知识图谱已保存: %s (实体=%d 关系=%d)", target, len(self.entities), len(self.relations))
        return target

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraphStore":
        """从 JSON 加载；文件不存在时返回空图谱而不是抛异常。"""
        if not path or not os.path.exists(path):
            logger.info("图谱文件不存在，返回空图谱: %s", path)
            store = cls(persist_path=path)
            return store
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls.from_dict(data)
        store.persist_path = path
        logger.info("知识图谱已加载: %s (实体=%d 关系=%d)", path, len(store.entities), len(store.relations))
        return store

    def clear(self) -> None:
        """清空图谱（重建场景）。"""
        with self._lock:
            self.entities.clear()
            self.relations.clear()
            self._adj.clear()
            self._edge_types.clear()
