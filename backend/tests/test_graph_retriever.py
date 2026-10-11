"""图谱检索服务的单元测试

重点验证三件事：
1. **多跳关系链真的进入了给 LLM 的上下文** —— 这是 GraphRAG 的增量所在
2. **全局聚合问题的答案真的出现在上下文里** —— 这是向量检索原理上做不到的
3. **上下文里保留了字符区间与原文证据** —— 保证前端既有的溯源高亮能复用

全部离线运行，不需要任何 API Key。
"""

import pytest

from app.services.graph_retriever import (
    GraphRetriever,
    is_global_query,
)
from app.services.graph_store import KnowledgeGraphStore, RelationTriple


# =============================================================================
# 问题类型判别
# =============================================================================


class TestIsGlobalQuery:
    """全局/局部问题的分流是 GraphRAG 的第一步决策。"""

    @pytest.mark.parametrize("query", [
        "文档里共涉及哪些机构？",
        "一共提到了几个患者？",
        "整体上讲了什么内容",
        "总结一下主要发现",
        "统计一下各类型实体的数量",
        "都包括哪些药物",
    ])
    def test_global_queries_detected(self, query):
        assert is_global_query(query) is True

    @pytest.mark.parametrize("query", [
        "张建国的血压是多少？",
        "沙库巴曲缬沙坦钠片的用法用量",
        "扩张型心肌病的诊断依据",
        "患者是什么时候入院的？",
        "这个病怎么治",
        "为什么要用华法林",
        "",
    ])
    def test_local_queries_not_global(self, query):
        assert is_global_query(query) is False

    @pytest.mark.parametrize("query", [
        # 同样含"多少"，但问的是聚合数量而不是取值 —— 必须和上面对照
        "一共有多少家机构",
        "提到了多少个患者",
        "共涉及多少种药物",
    ])
    def test_aggregate_count_questions_are_global(self, query):
        assert is_global_query(query) is True

    def test_value_question_vs_count_question_disambiguated(self):
        """同一句式里的"多少"，靠量词与取值词区分。

        这是实际踩到的坑：最初把 `多少` 直接当全局关键词，
        结果"血压是多少"被判成聚合问题，去读社区摘要，答案自然不对。
        """
        assert is_global_query("血压是多少") is False
        assert is_global_query("一共多少个指标") is True


# =============================================================================
# 测试用图谱
# =============================================================================


@pytest.fixture
def retriever():
    """构造两个社区的知识图谱：

    社区 A（医疗）：张建国 -患有-> 扩张型心肌病 -治疗-> 沙库巴曲缬沙坦钠片 -研发-> 诺华
                   扩张型心肌病 -表现-> 双下肢水肿
    社区 B（机构）：北京协和医院 -隶属于-> 卫健委

    两社区之间只有一条弱桥，便于验证社区发现与全局检索。
    """
    store = KnowledgeGraphStore()

    entities = [
        ("患者", "张建国", 0),
        ("诊断", "扩张型心肌病", 20),
        ("药物", "沙库巴曲缬沙坦钠片", 60),
        ("机构", "诺华", 100),
        ("症状", "双下肢水肿", 130),
        ("机构", "北京协和医院", 160),
        ("机构", "卫健委", 200),
    ]
    store.add_extractions(
        "doc-1", "病历.pdf",
        [
            {
                "extraction_class": etype,
                "extraction_text": name,
                "attributes": {"类型": etype},
                "char_interval": {"start_pos": pos, "end_pos": pos + len(name)},
            }
            for etype, name, pos in entities
        ],
    )

    store.add_triples(
        [
            RelationTriple("张建国", "患有", "扩张型心肌病",
                           evidence="诊断为扩张型心肌病", confidence=0.9, start_pos=40, end_pos=49),
            RelationTriple("扩张型心肌病", "治疗", "沙库巴曲缬沙坦钠片",
                           evidence="给予沙库巴曲缬沙坦钠片治疗", confidence=0.88),
            RelationTriple("沙库巴曲缬沙坦钠片", "研发", "诺华",
                           evidence="诺华研发的沙库巴曲缬沙坦钠", confidence=0.85),
            RelationTriple("扩张型心肌病", "表现", "双下肢水肿",
                           evidence="双下肢凹陷性水肿", confidence=0.8),
            RelationTriple("北京协和医院", "隶属于", "卫健委",
                           evidence="国家卫生健康委员会直属", confidence=0.9),
        ],
        doc_id="doc-1",
        doc_title="病历.pdf",
        auto_create_entities=False,
    )
    return GraphRetriever(store, default_hops=3, max_nodes=30, max_chains=15)


# =============================================================================
# 局部检索
# =============================================================================


class TestLocalRetrieval:
    def test_seed_linking(self, retriever):
        result = retriever.retrieve("张建国患了什么病")
        assert result.hit is True
        assert result.seeds[0][0].name == "张建国"
        assert result.seeds[0][1] == 1.0, "精确包含匹配置信度应为 1.0"

    def test_multi_hop_chain_reaches_far_entity(self, retriever):
        """核心断言：3 跳内必须能把"诺华"带进上下文。

        向量检索要召回"诺华"，依赖的是某个文本块恰好同时语义接近"张建国"，
        这在原理上不可靠；图谱检索通过关系路径保证可达。
        """
        result = retriever.retrieve("张建国用的药是哪家公司研发的", hops=3)

        assert "诺华" in result.context_text, "跨块实体必须出现在上下文里"
        assert any("诺华" in chain["text"] for chain in result.chains)

    def test_chain_text_is_human_readable(self, retriever):
        result = retriever.retrieve("张建国", hops=3)
        first = result.chains[0]["text"]
        assert "--[" in first, "关系链必须是 A --[关系]--> B 的可读形式"

    def test_no_match_returns_empty_context(self, retriever):
        result = retriever.retrieve("完全不存在的实体名称内容")
        assert result.hit is False
        assert result.context_text == ""

    def test_context_contains_entity_attributes(self, retriever):
        result = retriever.retrieve("张建国", hops=2)
        assert "相关实体" in result.context_text
        assert "类型=患者" in result.context_text or "患者" in result.context_text

    def test_context_contains_char_intervals_for_tracing(self, retriever):
        """上下文必须带字符区间，否则前端无法把答案里的实体高亮回原文。"""
        result = retriever.retrieve("张建国", hops=1)
        assert "字符" in result.context_text, "缺少溯源锚点会让前端高亮失效"

    def test_context_contains_relation_evidence(self, retriever):
        result = retriever.retrieve("张建国", hops=2)
        assert "关系证据" in result.context_text
        assert "诊断为扩张型心肌病" in result.context_text

    def test_empty_graph_is_safe(self):
        retriever = GraphRetriever(KnowledgeGraphStore())
        result = retriever.retrieve("任何问题")
        assert result.hit is False

    def test_empty_query_is_safe(self, retriever):
        assert retriever.retrieve("").hit is False

    def test_retrieve_for_explicit_entities(self, retriever):
        result = retriever.retrieve_for_entities(["沙库巴曲缬沙坦钠片"], hops=2)
        assert result.hit is True
        assert "诺华" in result.context_text, "显式指定实体也应做多跳扩展"

    def test_retrieve_for_unknown_entity(self, retriever):
        assert retriever.retrieve_for_entities(["不存在的实体"]).hit is False


class TestWeakEdgeMarking:
    """共现边必须被显式标注为弱证据，否则 LLM 会把共现当因果。"""

    @pytest.fixture
    def retriever_with_cooccurrence(self):
        store = KnowledgeGraphStore()
        store.add_extractions(
            "d1", "x.pdf",
            [
                {"extraction_class": "A", "extraction_text": "甲实体",
                 "char_interval": {"start_pos": 0, "end_pos": 3}},
                {"extraction_class": "B", "extraction_text": "乙实体",
                 "char_interval": {"start_pos": 4, "end_pos": 7}},
            ],
        )
        store.add_triples(
            [RelationTriple("甲实体", "共现", "乙实体", evidence="甲实体和乙实体一起出现",
                            confidence=0.35, start_pos=0, end_pos=11)],
            doc_id="d1", doc_title="x.pdf", auto_create_entities=False,
        )
        return GraphRetriever(store)

    def test_cooccurrence_flagged_in_context(self, retriever_with_cooccurrence):
        result = retriever_with_cooccurrence.retrieve("甲实体", hops=1)
        assert "共现" in result.context_text
        assert "不能作为关系成立的依据" in result.context_text, "必须明确警告 LLM 不要据此下结论"

    def test_cooccurrence_not_listed_as_strong_evidence(self, retriever_with_cooccurrence):
        result = retriever_with_cooccurrence.retrieve("甲实体", hops=1)
        assert "关系证据" not in result.context_text


class TestTruncation:
    def test_truncation_noted_in_context(self):
        """子图被截断时必须告知 LLM，否则它会以为这就是全貌。"""
        store = KnowledgeGraphStore()
        triples = [RelationTriple("中心", "连接", f"叶子{i}") for i in range(30)]
        store.add_triples(triples, doc_id="d1")

        retriever = GraphRetriever(store, max_nodes=8)
        result = retriever.retrieve("中心", hops=1)

        assert result.truncated is True
        assert "截断" in result.context_text


# =============================================================================
# 全局检索
# =============================================================================


class TestGlobalRetrieval:
    def test_communities_detected_and_summarized(self, retriever):
        result = retriever.retrieve_global("整体涉及哪些机构")
        assert result.mode == "global"
        assert len(result.communities) >= 1
        assert "社区摘要" in result.context_text

    def test_aggregate_answer_present_in_context(self, retriever):
        """核心断言：机构清单必须出现在上下文里。

        这正面回答了"为什么需要 GraphRAG"——向量检索把文档切块后，
        没有任何一块包含完整的机构清单，模型无法数出正确答案；
        社区摘要把全量成员一次列清。
        """
        result = retriever.retrieve_global("共涉及哪些机构")
        text = result.context_text

        assert "机构" in text
        # 三个机构实体都应在上下文中出现（成员清单或核心实体里）
        for name in ["诺华", "北京协和医院", "卫健委"]:
            assert name in text, f"聚合型上下文缺少 {name}"

    def test_entity_type_distribution_included(self, retriever):
        result = retriever.retrieve_global("统计各类实体数量")
        assert "实体类型分布" in result.context_text

    def test_relevance_ranking_puts_related_community_first(self, retriever):
        result = retriever.retrieve_global("张建国的诊断和药物")
        assert len(result.communities) >= 1
        # 最相关的社区里应含"张建国"
        top_members = [m["name"] for m in result.communities[0]["members"]]
        assert "张建国" in top_members

    def test_max_communities_limit(self, retriever):
        retriever.max_communities = 1
        result = retriever.retrieve_global("任意问题")
        assert len(result.communities) <= 1

    def test_empty_graph_is_safe(self):
        result = GraphRetriever(KnowledgeGraphStore()).retrieve_global("问题")
        assert result.hit is False and result.context_text == ""

    def test_global_summary_is_deterministic(self, retriever):
        """社区摘要必须可复现——否则评测数字每次跑都不一样，失去意义。"""
        first = retriever.retrieve_global("涉及哪些机构").context_text
        second = retriever.retrieve_global("涉及哪些机构").context_text
        assert first == second


# =============================================================================
# 自动分流
# =============================================================================


class TestSearchRouting:
    def test_global_query_routes_to_global(self, retriever):
        result = retriever.search("共涉及哪些机构")
        assert result.mode == "global"

    def test_local_query_routes_to_local(self, retriever):
        result = retriever.search("张建国患了什么病")
        assert result.mode == "local"
        assert result.hit is True

    def test_global_query_without_relevant_community_falls_back(self, retriever):
        """全局问题但社区都不相关时，应退化为局部检索，而不是给空上下文。"""
        result = retriever.search("一共提到了几个张建国")
        assert result.hit is True

    def test_local_query_with_no_entity_hit(self, retriever):
        assert retriever.search("毫无关联的问题内容").hit is False


# =============================================================================
# 序列化
# =============================================================================


class TestSerialization:
    def test_to_dict_shape(self, retriever):
        payload = retriever.retrieve("张建国", hops=2).to_dict()
        for key in [
            "query", "mode", "hops", "hit", "truncated",
            "seed_count", "node_count", "edge_count", "chain_count",
            "community_count", "seeds", "context_text",
        ]:
            assert key in payload, f"缺少字段 {key}"

    def test_seeds_serialized_with_confidence(self, retriever):
        payload = retriever.retrieve("张建国").to_dict()
        assert payload["seeds"][0]["name"] == "张建国"
        assert 0 < payload["seeds"][0]["link_confidence"] <= 1.0
