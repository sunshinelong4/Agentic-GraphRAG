"""关系抽取服务 - LLM 驱动的知识三元组抽取

解决的问题
----------
改造前，"知识图谱"的边是这样来的（`frontend/src/components/steps/KnowledgeGraph.tsx`
的 `generateRelationshipsFromItems()`）：

    if 人物 and 机构: relations.push({from: person, to: org, type: "属于"})
    if 人物 and 疾病: relations.push({from: person, to: disease, type: "患有"})

这是**笛卡尔积式硬编码**：任意一个"人物"和任意一个"机构"都会被连上一条"属于"，
不管它们在原文里有没有关系。后果是图里全是假边，检索价值为负。

本模块用 LLM 抽取真正的 (头实体, 关系, 尾实体) 三元组，并施加四道约束把幻觉压到最低：

    约束 1  实体锚定：把已知实体列表喂给 LLM，要求端点**必须**从中选取；
            抽完后再校验一遍，端点不在已知集合内的一律丢弃。
    约束 2  证据必填：每条关系必须给出原文证据片段，且该片段必须是原文的子串；
            校验不过时尝试用"包含头实体的那句话"兜底，兜底也失败则丢弃。
    约束 3  关系规范化：关系词长度与格式受限，避免 LLM 写出一整句话当关系名。
    约束 4  共现降级：LLM 调用失败或抽空时，退化为"同一段落内共现"的低置信度边，
            保证图谱不会因为一次 API 抖动就变成空图。

为什么要单独抽关系，而不是复用 `Extractor` 的"关系描述"类别：
`Extractor` 把关系当成一个**文本片段**抽取（extraction_text 是一整句话），
还需要下游再从 attributes 里猜主体1/主体2。本模块直接产出结构化的三元组，
并且只把"关系"当作一次针对性推理任务，提示词更短、约束更强、结果直接可落图。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.graph_store import (
    KnowledgeGraphStore,
    RelationTriple,
    normalize_entity_name,
)

logger = logging.getLogger(__name__)


# 关系词长度上限：超过这个长度说明 LLM 把整句话当关系名了
MAX_RELATION_LENGTH = 12

# 单段落最多参与共现配对的实体数，防止 n² 爆炸
MAX_COOCCURRENCE_ENTITIES = 8

# 关系抽取的系统提示词
RELATION_SYSTEM_PROMPT = """你是一个知识图谱关系抽取专家。你的任务是从文本中抽取实体之间的**明确关系**。

严格要求：
1. 只能使用【允许的实体列表】中的实体作为关系两端，不得创造新实体。
2. 只抽取文本中**明确陈述**的关系。如果两个实体只是同时出现但没有明确关系，不要抽取。
3. 每条关系必须给出**原文证据**，证据必须是原文的精确子串（不要改写、不要概括）。
4. 关系词要简短（2-6 个字），例如：患有、治疗、属于、研发、投资、任职、位于、导致。
5. 不要抽取自环（头尾实体相同）。

输出格式：只输出一个 JSON 数组，不要任何解释文字。数组元素格式：
[
  {"head": "头实体", "relation": "关系词", "tail": "尾实体", "evidence": "原文证据片段", "confidence": 0.9}
]

如果文本中不存在明确关系，输出空数组 []。"""


class RelationExtractor:
    """从文本中抽取关系三元组。

    依赖注入：`llm` 参数可以是任何具备 `.invoke(messages) -> str | AIMessage` 的对象。
    这让单元测试可以传入 FakeLLM，**无需真实 API Key、零成本、毫秒级**。
    """

    def __init__(
        self,
        llm: Any = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.last_stats: Dict[str, int] = {}

    # -------------------------------------------------------------------------
    # LLM 接入
    # -------------------------------------------------------------------------

    @property
    def llm(self) -> Any:
        """惰性创建 LLM，避免在没有 API Key 的环境下 import 就崩。"""
        if self._llm is None:
            import os

            from langchain_openai import ChatOpenAI

            model = self.model or os.getenv("DEFAULT_MODEL", "deepseek-chat")
            key = self.api_key or os.getenv("DEEPSEEK_API_KEY", "")
            if not key:
                raise ValueError("未配置 DEEPSEEK_API_KEY，无法进行关系抽取")
            self._llm = ChatOpenAI(
                model=model,
                api_key=key,
                base_url=self.base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                temperature=self.temperature,
                timeout=45,
                max_retries=1,
            )
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 并统一取出文本内容。"""
        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            # 部分模型返回分块内容，拼起来
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content or "")

    # -------------------------------------------------------------------------
    # 主流程
    # -------------------------------------------------------------------------

    def extract(
        self,
        text: str,
        known_entities: Sequence[str],
        max_relations: int = 60,
        use_cooccurrence_fallback: bool = True,
    ) -> List[RelationTriple]:
        """从文本中抽取关系三元组。

        Args:
            text: 原文
            known_entities: 已知实体名列表（通常来自 `Extractor` 的抽取结果）。
                同时作为**实体锚定白名单**使用。
            max_relations: 关系数上限
            use_cooccurrence_fallback: LLM 无产出时是否用共现边兜底

        Returns:
            RelationTriple 列表（含 evidence 与置信度）
        """
        self.last_stats = {
            "paragraphs_total": 0,
            "paragraphs_processed": 0,
            "raw_triples": 0,
            "accepted": 0,
            "rejected_unanchored": 0,
            "rejected_self_loop": 0,
            "llm_errors": 0,
            "fallback_used": 0,
        }

        if not text or not known_entities:
            return []

        # 规范化后的已知实体集合（端点校验的白名单）
        anchor_set = {normalize_entity_name(n) for n in known_entities}
        anchor_set = {n for n in anchor_set if n}
        if not anchor_set:
            return []

        # 规范化名 -> 原文写法（用于回填 head/tail，保证图谱里的展示名和抽取结果一致）
        display_name: Dict[str, str] = {}
        for name in known_entities:
            nid = normalize_entity_name(name)
            if nid and nid not in display_name:
                display_name[nid] = str(name).strip()

        paragraphs = _split_paragraphs(text)
        self.last_stats["paragraphs_total"] = len(paragraphs)

        results: List[RelationTriple] = []
        seen: set = set()

        for para_text, para_offset in paragraphs:
            if len(results) >= max_relations:
                break

            # 只处理至少含 2 个已知实体的段落 —— 其余段落不可能产出关系。
            # 这一步能省掉大量 token，长文档上通常是 70%+ 的段落被跳过。
            present = [eid for eid in anchor_set if eid in normalize_entity_name(para_text)]
            if len(present) < 2:
                continue

            self.last_stats["paragraphs_processed"] += 1

            triples, had_candidates = self._extract_from_paragraph(
                para_text=para_text,
                para_offset=para_offset,
                present_entities=present,
                display_name=display_name,
            )

            # 降级只在 LLM **没有产出**（调用失败 / 空响应 / 无法解析）时触发。
            # 如果 LLM 给了候选、但被约束全部拦下，那是约束**正常工作**的结果，
            # 此时再补共现边等于"检测到幻觉后用兜底策略把噪声塞回来"，
            # 会让防幻觉机制形同虚设。这个边界是靠测试发现的。
            if not triples and not had_candidates and use_cooccurrence_fallback:
                triples = self._cooccurrence_fallback(
                    para_text, para_offset, present, display_name
                )
                if triples:
                    self.last_stats["fallback_used"] += 1

            for triple in triples:
                key = (
                    normalize_entity_name(triple.head),
                    triple.relation,
                    normalize_entity_name(triple.tail),
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append(triple)
                if len(results) >= max_relations:
                    break

        self.last_stats["accepted"] = len(results)
        logger.info(
            "关系抽取完成: 段落 %d/%d 处理，接受 %d 条关系（LLM错误 %d，降级 %d 段）",
            self.last_stats["paragraphs_processed"],
            self.last_stats["paragraphs_total"],
            len(results),
            self.last_stats["llm_errors"],
            self.last_stats["fallback_used"],
        )
        return results

    # -------------------------------------------------------------------------
    # 单段落抽取
    # -------------------------------------------------------------------------

    def _extract_from_paragraph(
        self,
        para_text: str,
        para_offset: int,
        present_entities: Sequence[str],
        display_name: Dict[str, str],
    ) -> Tuple[List[RelationTriple], bool]:
        """对单个段落做一次 LLM 关系抽取 + 校验。

        Returns:
            (通过校验的三元组, LLM 是否产出了可解析的候选)。
            第二个返回值决定调用方是否允许走共现降级：只有在 LLM
            **没有产出**时才降级；约束主动拦截的情况不算"没产出"。
        """
        user_prompt = (
            "【允许的实体列表】\n"
            + "、".join(display_name.get(eid, eid) for eid in sorted(present_entities))
            + f"\n\n【待抽取文本】\n{para_text}\n\n请输出 JSON 数组："
        )

        try:
            raw = self._call_llm(RELATION_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # 网络/额度/解析异常都不应让整条链路挂掉
            logger.warning("关系抽取 LLM 调用失败，将走降级策略: %s", exc)
            self.last_stats["llm_errors"] += 1
            return [], False

        candidates = _parse_json_array(raw)
        self.last_stats["raw_triples"] += len(candidates)
        if not candidates:
            return [], False

        validated = self._validate(
            candidates=candidates,
            anchor_set=set(present_entities),
            display_name=display_name,
            para_text=para_text,
            para_offset=para_offset,
        )
        return validated, True

    def _validate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor_set: set,
        display_name: Dict[str, str],
        para_text: str,
        para_offset: int,
    ) -> List[RelationTriple]:
        """四道约束的校验：锚定 / 自环 / 关系词规范 / 证据可验证。"""
        accepted: List[RelationTriple] = []
        para_norm = normalize_entity_name(para_text)

        for item in candidates:
            if not isinstance(item, dict):
                continue

            head_id = normalize_entity_name(item.get("head") or "")
            tail_id = normalize_entity_name(item.get("tail") or "")
            if not head_id or not tail_id:
                continue

            # 约束 1：实体锚定 —— 端点必须在本段的已知实体集合内
            if head_id not in anchor_set or tail_id not in anchor_set:
                self.last_stats["rejected_unanchored"] += 1
                continue
            # 端点必须真的出现在本段文本里（防止 LLM 从其它段落"借"实体）
            if head_id not in para_norm or tail_id not in para_norm:
                self.last_stats["rejected_unanchored"] += 1
                continue

            # 约束 3：拒绝自环
            if head_id == tail_id:
                self.last_stats["rejected_self_loop"] += 1
                continue

            relation = _normalize_relation(item.get("relation"))
            if not relation:
                continue

            # 约束 2：证据必须是原文子串；否则用"包含头实体的那句话"兜底
            evidence = str(item.get("evidence") or "").strip()
            confidence = _safe_confidence(item.get("confidence"))

            if evidence and evidence in para_text:
                start = para_offset + para_text.find(evidence)
                end = start + len(evidence)
            else:
                # 兜底：找包含头实体的句子作为证据
                fallback_sentence = _sentence_containing(para_text, display_name.get(head_id, head_id))
                if fallback_sentence:
                    evidence = fallback_sentence
                    confidence = min(confidence, 0.6)  # 证据是推断来的，降置信度
                    start = para_offset + para_text.find(fallback_sentence)
                    end = start + len(fallback_sentence)
                else:
                    evidence = ""
                    confidence = min(confidence, 0.5)
                    start = end = None

            accepted.append(
                RelationTriple(
                    head=display_name.get(head_id, head_id),
                    relation=relation,
                    tail=display_name.get(tail_id, tail_id),
                    head_type="",
                    tail_type="",
                    evidence=evidence,
                    confidence=confidence,
                    start_pos=start,
                    end_pos=end,
                )
            )

        return accepted

    # -------------------------------------------------------------------------
    # 共现降级
    # -------------------------------------------------------------------------

    def _cooccurrence_fallback(
        self,
        para_text: str,
        para_offset: int,
        present_entities: Sequence[str],
        display_name: Dict[str, str],
    ) -> List[RelationTriple]:
        """降级策略：同一段落内共现的实体之间建立低置信度"共现"边。

        这不是"假边"——共现是弱证据，故标 0.35 置信度并在关系名上明确区分。
        好处是：图谱永远不会因为一次 LLM 抖动变成空图，检索退化成
        "段落级共现检索"，仍比完全没有图信息强。
        """
        # 按在段落中出现的先后排序，并限量，避免 n² 爆炸
        ordered = sorted(
            (eid for eid in present_entities if eid in para_text),
            key=lambda eid: para_text.find(eid),
        )[:MAX_COOCCURRENCE_ENTITIES]

        sentence_cache: Dict[str, str] = {}
        triples: List[RelationTriple] = []

        for i, head_id in enumerate(ordered):
            for tail_id in ordered[i + 1:]:
                if head_id == tail_id:
                    continue
                if head_id not in sentence_cache:
                    sentence_cache[head_id] = _sentence_containing(
                        para_text, display_name.get(head_id, head_id)
                    )
                evidence = sentence_cache[head_id] or para_text[:120]
                # 证据是段落内的句子，位置可以精确定位到全文
                local_start = para_text.find(evidence)
                start = para_offset + local_start if local_start != -1 else None
                end = start + len(evidence) if start is not None else None
                triples.append(
                    RelationTriple(
                        head=display_name.get(head_id, head_id),
                        relation="共现",
                        tail=display_name.get(tail_id, tail_id),
                        evidence=evidence,
                        confidence=0.35,
                        start_pos=start,
                        end_pos=end,
                    )
                )
        return triples


# =============================================================================
# 纯函数工具（无副作用，单独测试）
# =============================================================================


def _split_paragraphs(text: str) -> List[Tuple[str, int]]:
    """按空行切段，返回 [(段落文本, 在全文中的起始偏移)]。

    保留偏移量是为了能把关系证据映射回**全文**的字符区间，
    这样关系也能像实体一样被溯源与高亮。
    """
    paragraphs: List[Tuple[str, int]] = []
    for match in re.finditer(r"[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*", text):
        para = match.group(0).strip()
        if para:
            # 重新定位去空白后的真实起点
            offset = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            paragraphs.append((para, offset))
    return paragraphs


def _sentence_containing(text: str, needle: str, max_length: int = 200) -> str:
    """找出包含 needle 的那句话。用于给关系生成可读证据。"""
    if not needle:
        return ""
    idx = text.find(needle)
    if idx == -1:
        return ""
    # 向左找句末标点
    start = idx
    while start > 0 and text[start - 1] not in "。！？\n；;.!?":
        start -= 1
    end = idx + len(needle)
    while end < len(text) and text[end] not in "。！？\n；;.!?":
        end += 1
    if end < len(text):
        end += 1
    sentence = text[start:end].strip()
    return sentence[:max_length]


def _normalize_relation(value: Any) -> str:
    """规范化关系词：去空白、去标点、限长。无效时返回空串。"""
    if value is None:
        return ""
    relation = re.sub(r"\s+", "", str(value)).strip("，。、：:;；-—>→")
    if not relation or len(relation) > MAX_RELATION_LENGTH:
        return ""
    # 纯符号不算关系
    if not any(ch.isalnum() for ch in relation):
        return ""
    return relation


def _safe_confidence(value: Any, default: float = 0.8) -> float:
    """把 LLM 给的置信度夹到 [0.1, 1.0]，非法值回退默认值。"""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf <= 0:
        return default
    return max(0.1, min(1.0, conf))


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    """从 LLM 输出里稳健地取出 JSON 数组。

    要处理的脏情况：
    - 被 ```json ... ``` 代码块包裹
    - 前后带解释文字（"好的，以下是抽取结果：..."）
    - 返回的是 {"relations": [...]} 而不是裸数组
    - 完全不是 JSON

    策略：先去 fence，再截取第一个 '[' 到最后一个 ']'；失败则尝试解析成对象并找数组字段。
    """
    if not raw:
        return []

    text = raw.strip()

    # 去掉 markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 直接就是数组
    if text.startswith("["):
        parsed = _try_loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    # 截取第一个 [ 到最后一个 ]
    left, right = text.find("["), text.rfind("]")
    if left != -1 and right > left:
        parsed = _try_loads(text[left:right + 1])
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    # 可能是 {"relations": [...]} 形式的对象
    left, right = text.find("{"), text.rfind("}")
    if left != -1 and right > left:
        parsed = _try_loads(text[left:right + 1])
        if isinstance(parsed, dict):
            for key in ("relations", "triples", "result", "data", "关系"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]

    logger.debug("关系抽取输出无法解析为 JSON 数组: %s", raw[:200])
    return []


def _try_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# =============================================================================
# 便捷入口：抽取 + 落图 一步到位
# =============================================================================


def build_graph_from_extractions(
    store: KnowledgeGraphStore,
    extractor: RelationExtractor,
    text: str,
    doc_id: str,
    doc_title: str,
    extractions: Sequence[Dict[str, Any]],
    max_relations: int = 60,
) -> Dict[str, Any]:
    """把一个文档的抽取结果 + 原文，完整落成图谱。

    这是 API 层调用的主入口，串起：
        实体入库（带字符区间，可溯源） -> 关系抽取 -> 关系入库

    Args:
        store: 图谱存储
        extractor: 关系抽取器
        text: 原文（用于关系证据定位）
        doc_id / doc_title: 文档标识
        extractions: `Extractor.extract()` 的 extractions
        max_relations: 关系数上限

    Returns:
        落图统计
    """
    entity_stats = store.add_extractions(doc_id, doc_title, extractions)

    entity_names = [
        ext.get("extraction_text", "")
        for ext in extractions
        if (ext.get("extraction_text") or "").strip()
    ]

    triples = extractor.extract(text=text, known_entities=entity_names, max_relations=max_relations)

    # 实体已经通过 add_extractions 建好（带位置），关系只补边不重复造实体
    relation_stats = store.add_triples(
        triples, doc_id=doc_id, doc_title=doc_title, auto_create_entities=False
    )

    return {
        "doc_id": doc_id,
        "doc_title": doc_title,
        **entity_stats,
        **relation_stats,
        "relation_extraction": dict(extractor.last_stats),
        "graph_stats": store.stats(),
    }
