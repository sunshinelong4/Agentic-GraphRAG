"""关系抽取服务的单元测试

核心手法：**依赖注入 FakeLLM**。

`RelationExtractor` 的 `llm` 参数接受任何具备 `.invoke(messages)` 的对象，
因此这里注入一个脚本化的假 LLM，就能在没有 API Key、不花一分钱、
毫秒级跑完的情况下，验证：

- 四道防幻觉约束（实体锚定 / 证据可验证 / 关系词规范 / 共现降级）是否真的生效
- 脏输出（代码围栏、前后带解释文字、包成对象、完全不是 JSON）能否稳健解析
- 字符区间是否正确映射回全文（关系级溯源）

这也是"可测试性"反过来影响设计的一个例子：正是为了让这里能注入假 LLM，
`RelationExtractor` 才没有把 `ChatOpenAI` 硬编码在构造函数里。
"""

import pytest

from app.services.graph_store import KnowledgeGraphStore, normalize_entity_name
from app.services.relation_extractor import (
    RelationExtractor,
    _normalize_relation,
    _parse_json_array,
    _safe_confidence,
    _sentence_containing,
    _split_paragraphs,
    build_graph_from_extractions,
)


class FakeLLM:
    """脚本化假 LLM：按顺序吐出预设响应，并记录收到的 prompt 便于断言。"""

    def __init__(self, responses):
        # 允许传单个字符串或字符串列表
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else [responses]
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            return _Resp("[]")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    @property
    def call_count(self):
        return len(self.calls)

    @property
    def last_user_prompt(self):
        return self.calls[-1][-1]["content"] if self.calls else ""


class _Resp:
    """模拟 LangChain AIMessage：只需要 `.content` 属性。"""

    def __init__(self, content):
        self.content = content


# =============================================================================
# 纯函数：JSON 解析
# =============================================================================


class TestParseJsonArray:
    """LLM 输出从来不会干净，解析器必须能扛住各种脏格式。"""

    def test_bare_array(self):
        raw = '[{"head": "A", "relation": "患有", "tail": "B"}]'
        assert _parse_json_array(raw) == [{"head": "A", "relation": "患有", "tail": "B"}]

    def test_fenced_json(self):
        raw = '```json\n[{"head": "A", "relation": "r", "tail": "B"}]\n```'
        assert len(_parse_json_array(raw)) == 1

    def test_fenced_without_language_tag(self):
        raw = '```\n[{"head": "A", "relation": "r", "tail": "B"}]\n```'
        assert len(_parse_json_array(raw)) == 1

    def test_prose_before_and_after(self):
        """LLM 经常在 JSON 前后加一句解释。"""
        raw = '好的，以下是抽取结果：\n[{"head": "A", "relation": "r", "tail": "B"}]\n希望有帮助！'
        assert len(_parse_json_array(raw)) == 1

    def test_object_wrapper(self):
        raw = '{"relations": [{"head": "A", "relation": "r", "tail": "B"}]}'
        assert len(_parse_json_array(raw)) == 1

    def test_empty_array(self):
        assert _parse_json_array("[]") == []

    @pytest.mark.parametrize("raw", ["", "   ", "完全不是 JSON", "{坏掉的 json", None])
    def test_garbage_returns_empty(self, raw):
        assert _parse_json_array(raw) == []

    def test_non_dict_items_filtered(self):
        """数组里混入字符串/数字要能过滤掉，而不是让后续 .get 崩掉。"""
        raw = '[{"head": "A", "relation": "r", "tail": "B"}, "垃圾", 42, null]'
        parsed = _parse_json_array(raw)
        assert len(parsed) == 1 and parsed[0]["head"] == "A"


class TestNormalizeRelation:
    def test_strips_punctuation_and_space(self):
        assert _normalize_relation(" 患有 。") == "患有"

    def test_too_long_rejected(self):
        """关系词超过 12 字说明 LLM 把整句话当关系名了，必须拒。"""
        assert _normalize_relation("这是一个非常非常长的关系描述句子") == ""

    def test_pure_symbols_rejected(self):
        assert _normalize_relation("——>") == ""

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_empty_rejected(self, value):
        assert _normalize_relation(value) == ""


class TestSafeConfidence:
    @pytest.mark.parametrize("value,expected", [
        (0.9, 0.9), (1.5, 1.0), (-1, 0.8), ("0.7", 0.7),
        ("abc", 0.8), (None, 0.8), (0, 0.8),
    ])
    def test_clamped(self, value, expected):
        assert _safe_confidence(value) == pytest.approx(expected)


class TestSentenceContaining:
    def test_extracts_sentence(self):
        text = "既往史：高血压15年。患者张三被诊断为扩张型心肌病。治疗上给予呋塞米。"
        sentence = _sentence_containing(text, "扩张型心肌病")
        assert "扩张型心肌病" in sentence
        assert "呋塞米" not in sentence, "不能跨句边界"

    def test_missing_needle_returns_empty(self):
        assert _sentence_containing("无关文本", "找不到的实体") == ""


class TestSplitParagraphs:
    def test_splits_on_blank_line(self):
        paras = _split_paragraphs("第一段内容。\n\n第二段内容。")
        assert [p[0] for p in paras] == ["第一段内容。", "第二段内容。"]

    def test_offsets_are_correct(self):
        """偏移量必须指向全文真实位置，否则关系级溯源会错位。"""
        text = "第一段内容。\n\n第二段内容。"
        paras = _split_paragraphs(text)
        for para, offset in paras:
            assert text[offset:offset + len(para)] == para

    def test_single_newline_same_paragraph(self):
        """单换行不分段（Markdown 里的软换行）。"""
        paras = _split_paragraphs("第一行\n第二行\n\n另起一段")
        assert len(paras) == 2
        assert "\n" in paras[0][0]


# =============================================================================
# 四种约束
# =============================================================================


TEXT = "患者张三，男，62岁。诊断为扩张型心肌病。给予沙库巴曲缬沙坦钠片治疗。"
ENTITIES = ["张三", "扩张型心肌病", "沙库巴曲缬沙坦钠片"]


def _extractor_with(response):
    return RelationExtractor(llm=FakeLLM(response))


class TestConstraint1_EntityAnchoring:
    """约束 1：端点必须来自已知实体列表，且真的出现在本段。"""

    def test_valid_triple_accepted(self):
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
            ' "evidence": "诊断为扩张型心肌病", "confidence": 0.9}]'
        )
        triples = extractor.extract(TEXT, ENTITIES)

        assert len(triples) == 1
        assert triples[0].head == "张三"
        assert triples[0].relation == "患有"
        assert triples[0].tail == "扩张型心肌病"

    def test_invented_entity_rejected(self):
        """LLM 凭空造一个实体当端点 -> 必须丢弃（这是幻觉的主要形式）。"""
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "虚构疾病XYZ", "evidence": "x"}]'
        )
        triples = extractor.extract(TEXT, ENTITIES)

        assert triples == []
        assert extractor.last_stats["rejected_unanchored"] == 1

    def test_known_entity_from_other_paragraph_rejected(self):
        """实体在全集里，但**不在本段**——说明是跨段硬凑的关系。"""
        text = "张三患有扩张型心肌病。\n\n沙库巴曲缬沙坦钠片是常用药物。"
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "服用", "tail": "沙库巴曲缬沙坦钠片", "evidence": "x"}]'
        )
        # 第 1 段只有 1 个实体（张三、扩张型心肌病 -> 其实是 2 个，会触发调用）
        triples = extractor.extract(text, ENTITIES)
        # 无论走哪条路径，都不允许出现跨段的"张三-沙库巴曲缬沙坦钠片"直接关系
        assert all(
            not (t.head == "张三" and t.tail == "沙库巴曲缬沙坦钠片") for t in triples
        )


class TestConstraint2_EvidenceVerifiable:
    """约束 2：证据必须是原文子串，否则用包含头实体的句子兜底并降置信度。"""

    def test_exact_evidence_positions_mapped_to_full_text(self):
        """关系级溯源：证据的字符区间必须能在全文里取出同样的文字。"""
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
            ' "evidence": "诊断为扩张型心肌病", "confidence": 0.9}]'
        )
        triple = extractor.extract(TEXT, ENTITIES)[0]

        assert triple.evidence == "诊断为扩张型心肌病"
        assert TEXT[triple.start_pos:triple.end_pos] == triple.evidence

    def test_paraphrased_evidence_falls_back_and_lowers_confidence(self):
        """证据被 LLM 改写了 -> 用包含头实体的原句兜底，置信度必须下调。"""
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
            ' "evidence": "张三罹患扩张型心肌病", "confidence": 0.95}]'
        )
        triple = extractor.extract(TEXT, ENTITIES)[0]

        assert triple.confidence <= 0.6, "推断出来的证据不能保持高置信度"
        assert triple.evidence in TEXT, "兜底证据仍必须是原文子串"
        assert TEXT[triple.start_pos:triple.end_pos] == triple.evidence


class TestConstraint3_NoSelfLoop:
    def test_self_loop_rejected(self):
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "是", "tail": "张三", "evidence": "患者张三"}]'
        )
        triples = extractor.extract(TEXT, ENTITIES)
        assert triples == []
        assert extractor.last_stats["rejected_self_loop"] == 1


class TestFallbackBoundary:
    """降级触发条件必须精确：只在 LLM **没有产出**时降级。

    这是一个由测试发现的真实设计缺陷：
    最初只要 `_validate` 返回空就降级，导致"LLM 幻觉 -> 约束拦下 -> 立刻补共现边"
    ——防幻觉机制被自己的兜底策略绕过了。正确语义是：
      - LLM 报错 / 返回空 / 输出无法解析  -> 没产出 -> 降级（保证图不空）
      - LLM 给了候选但全被约束拦下        -> 有产出 -> 不降级（约束生效，尊重结果）
    """

    def test_all_candidates_rejected_does_not_trigger_fallback(self):
        extractor = _extractor_with(
            '[{"head": "虚构实体A", "relation": "患有", "tail": "虚构实体B", "evidence": "x"}]'
        )
        triples = extractor.extract(TEXT, ENTITIES)

        assert triples == [], "约束拦下全部候选后不应补共现边"
        assert extractor.last_stats["rejected_unanchored"] == 1
        assert extractor.last_stats["fallback_used"] == 0, "有产出就不该降级"

    def test_self_loop_only_does_not_trigger_fallback(self):
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "是", "tail": "张三", "evidence": "患者张三"}]'
        )
        extractor.extract(TEXT, ENTITIES)
        assert extractor.last_stats["fallback_used"] == 0

    def test_unparseable_output_triggers_fallback(self):
        """输出完全不是 JSON -> 视为没产出 -> 允许降级。"""
        extractor = _extractor_with("抱歉，我无法完成这个任务。")
        triples = extractor.extract(TEXT, ENTITIES)
        assert all(t.relation == "共现" for t in triples)
        assert extractor.last_stats["fallback_used"] == 1


class TestConstraint4_CoooccurrenceFallback:
    """约束 4：LLM 挂了也不能让图谱变成空图。"""

    def test_llm_exception_triggers_fallback(self):
        extractor = RelationExtractor(llm=FakeLLM([RuntimeError("API 超时")]))
        triples = extractor.extract(TEXT, ENTITIES)

        assert len(triples) > 0, "API 抖动不应导致空图谱"
        assert all(t.relation == "共现" for t in triples)
        assert all(t.confidence == pytest.approx(0.35) for t in triples)
        assert extractor.last_stats["llm_errors"] == 1
        assert extractor.last_stats["fallback_used"] == 1

    def test_empty_llm_output_triggers_fallback(self):
        extractor = _extractor_with("[]")
        triples = extractor.extract(TEXT, ENTITIES)
        assert all(t.relation == "共现" for t in triples)

    def test_fallback_can_be_disabled(self):
        extractor = _extractor_with("[]")
        triples = extractor.extract(TEXT, ENTITIES, use_cooccurrence_fallback=False)
        assert triples == []

    def test_fallback_edges_are_low_confidence_and_marked(self):
        """共现边必须能被识别出来，不能和真关系混在一起。"""
        extractor = _extractor_with("[]")
        triples = extractor.extract(TEXT, ENTITIES)
        for triple in triples:
            assert triple.relation == "共现"
            assert triple.confidence < 0.5, "共现是弱证据，置信度必须显著低于 LLM 抽取"

    def test_fallback_positions_mapped(self):
        extractor = _extractor_with("[]")
        for triple in extractor.extract(TEXT, ENTITIES):
            if triple.start_pos is not None:
                assert TEXT[triple.start_pos:triple.end_pos] == triple.evidence


# =============================================================================
# 工程行为：省 token、去重、限额
# =============================================================================


class TestEfficiency:
    def test_paragraph_with_single_entity_skips_llm(self):
        """只有一个实体的段落不可能产出关系，必须跳过——这是省 token 的关键。"""
        text = "张三前来就诊。\n\n天气很好，阳光明媚。\n\n扩张型心肌病需要长期管理。"
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        extractor.extract(text, ["张三", "扩张型心肌病"])

        assert llm.call_count == 0, "三段各只有一个实体，不应产生任何 LLM 调用"
        assert extractor.last_stats["paragraphs_processed"] == 0

    def test_only_relevant_paragraphs_are_processed(self):
        text = (
            "张三患有扩张型心肌病。\n\n"          # 2 个实体 -> 处理
            "今天天气不错，适合出门散步。\n\n"      # 0 个实体 -> 跳过
            "沙库巴曲缬沙坦钠片可用于治疗。\n\n"    # 1 个实体 -> 跳过
            "张三服用了沙库巴曲缬沙坦钠片。"        # 2 个实体 -> 处理
        )
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        extractor.extract(text, ENTITIES)

        assert llm.call_count == 2
        assert extractor.last_stats["paragraphs_total"] == 4
        assert extractor.last_stats["paragraphs_processed"] == 2

    def test_duplicate_triples_deduplicated(self):
        """同一关系在多段被重复抽到，只保留一条（后续由图谱层累加置信度）。"""
        text = "张三患有扩张型心肌病。\n\n张三确诊扩张型心肌病。"
        response = ('[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
                    ' "evidence": "张三患有扩张型心肌病"}]')
        extractor = RelationExtractor(llm=FakeLLM([response, response]))
        triples = extractor.extract(text, ENTITIES)

        assert len(triples) == 1

    def test_max_relations_cap(self):
        entities = [f"实体{i}" for i in range(6)]
        text = "".join(entities) + "。"
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        triples = extractor.extract(text, entities, max_relations=3, use_cooccurrence_fallback=True)
        assert len(triples) <= 3

    def test_no_entities_returns_empty_without_calling_llm(self):
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        assert extractor.extract(TEXT, []) == []
        assert llm.call_count == 0

    def test_empty_text_returns_empty(self):
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        assert extractor.extract("", ENTITIES) == []
        assert llm.call_count == 0

    def test_prompt_contains_anchor_entity_list(self):
        """提示词里必须带上白名单，否则"实体锚定"约束在源头就失效了。"""
        llm = FakeLLM("[]")
        extractor = RelationExtractor(llm=llm)
        extractor.extract(TEXT, ENTITIES)

        prompt = llm.last_user_prompt
        for entity in ENTITIES:
            assert entity in prompt, f"白名单里缺少 {entity}"


# =============================================================================
# 端到端：抽取 + 落图
# =============================================================================


class TestBuildGraphEndToEnd:
    def test_full_pipeline(self):
        store = KnowledgeGraphStore()
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
            ' "evidence": "诊断为扩张型心肌病", "confidence": 0.9},'
            ' {"head": "扩张型心肌病", "relation": "治疗", "tail": "沙库巴曲缬沙坦钠片",'
            ' "evidence": "给予沙库巴曲缬沙坦钠片治疗", "confidence": 0.85}]'
        )
        extractions = [
            {"extraction_class": cls, "extraction_text": name,
             "char_interval": {"start_pos": TEXT.find(name), "end_pos": TEXT.find(name) + len(name)}}
            for cls, name in [("人物", "张三"), ("诊断", "扩张型心肌病"), ("药物", "沙库巴曲缬沙坦钠片")]
        ]

        result = build_graph_from_extractions(
            store=store, extractor=extractor, text=TEXT,
            doc_id="d1", doc_title="病历.pdf", extractions=extractions,
        )

        assert result["entities_added"] == 3
        assert result["relations_added"] == 2
        stats = store.stats()
        assert stats["entity_count"] == 3
        assert stats["relation_count"] == 2
        assert stats["document_count"] == 1
        assert stats["traceable_ratio"] == 1.0

        # 图上的多跳能力必须真的可用
        path = store.shortest_path("张三", "沙库巴曲缬沙坦钠片")
        assert path is not None and path["hops"] == 2

    def test_relations_do_not_duplicate_entities(self):
        """关系落图时不应重复创建实体（否则会污染 document_count 与溯源率）。"""
        store = KnowledgeGraphStore()
        extractor = _extractor_with(
            '[{"head": "张三", "relation": "患有", "tail": "扩张型心肌病",'
            ' "evidence": "诊断为扩张型心肌病"}]'
        )
        extractions = [
            {"extraction_class": "人物", "extraction_text": "张三",
             "char_interval": {"start_pos": 2, "end_pos": 4}},
            {"extraction_class": "诊断", "extraction_text": "扩张型心肌病",
             "char_interval": {"start_pos": 8, "end_pos": 14}},
        ]
        build_graph_from_extractions(
            store=store, extractor=extractor, text=TEXT,
            doc_id="d1", doc_title="病历.pdf", extractions=extractions,
        )

        assert store.stats()["entity_count"] == 2
        assert store.stats()["mention_count"] == 2, "关系不应额外产生 mention"
