"""知识图谱存储的单元测试

测试策略：
1. **纯离线**——不调用任何 LLM / 网络，全部用手工构造的数据，因此毫秒级可跑、零成本。
2. **断言行为而不是实现**——例如"多跳检索必须能到达 2 跳外的实体"，
   而不是断言内部 BFS 的访问顺序。
3. 关键用例是 `TestMultiHopReachability`：它证明图检索能回答
   **单个文本块里不存在答案**的问题，这正是 GraphRAG 相对向量 RAG 的价值所在。
"""

import json
import os

import pytest

from app.services.graph_store import (
    GraphRelation,
    KnowledgeGraphStore,
    Mention,
    RelationTriple,
    is_valid_entity_name,
    normalize_entity_name,
)


# =============================================================================
# 实体名规范化
# =============================================================================


class TestNormalization:
    """实体名规范化：决定"哪些写法算同一个实体"，是图谱质量的第一道关。"""

    def test_wrapper_chars_stripped(self):
        """书名号/引号包裹的同一实体应归并。"""
        assert normalize_entity_name("《王者荣耀》") == normalize_entity_name("王者荣耀")
        assert normalize_entity_name("「连连数字」") == normalize_entity_name("连连数字")
        assert normalize_entity_name('"腾讯控股"') == normalize_entity_name("腾讯控股")

    def test_fullwidth_to_halfwidth(self):
        """全角字符应折叠为半角，避免同一实体被拆成两个节点。"""
        assert normalize_entity_name("ＡＢＣ") == normalize_entity_name("ABC")

    def test_ascii_case_insensitive(self):
        """ASCII 大小写不敏感。"""
        assert normalize_entity_name("IBM") == normalize_entity_name("ibm")

    def test_whitespace_collapsed(self):
        """多余空白（含全角空格）应折叠。"""
        assert normalize_entity_name("  腾讯   控股 ") == normalize_entity_name("腾讯 控股")

    def test_chinese_preserved(self):
        """中文不能被破坏。"""
        assert normalize_entity_name("扩张型心肌病") == "扩张型心肌病"

    def test_empty_and_none(self):
        assert normalize_entity_name("") == ""
        assert normalize_entity_name(None) == ""


class TestEntityValidity:
    """哪些名字值得进图。单字实体（"的"、"了"）是噪声，必须挡掉。"""

    @pytest.mark.parametrize("name", ["张", "a", "", "   ", "——", "《》"])
    def test_rejects_noise(self, name):
        assert is_valid_entity_name(normalize_entity_name(name)) is False

    @pytest.mark.parametrize("name", ["张三", "扩张型心肌病", "AB", "1.5亿元"])
    def test_accepts_meaningful(self, name):
        assert is_valid_entity_name(normalize_entity_name(name)) is True


# =============================================================================
# 实体与关系的写入
# =============================================================================


class TestEntityUpsert:
    def test_alias_merge_and_mentions(self):
        """同一实体的不同写法应归并为一个节点，并累积提及（溯源锚点）。"""
        store = KnowledgeGraphStore()
        store.upsert_entity("《王者荣耀》", "产品", mention=Mention("d1", "财报", 10, 16, "《王者荣耀》"))
        store.upsert_entity("王者荣耀", "产品", mention=Mention("d1", "财报", 200, 204, "王者荣耀"))

        assert len(store.entities) == 1, "两种写法必须归并成一个节点"
        entity = store.get_entity("王者荣耀")
        assert entity.mention_count == 2
        assert entity.traceable_ratio == 1.0, "两次提及都带字符区间"

    def test_mention_without_position_lowers_traceable_ratio(self):
        """缺字符区间的提及会拉低可溯源率——这是评估抽取质量的指标。"""
        store = KnowledgeGraphStore()
        store.upsert_entity("张三", "人物", mention=Mention("d1", "病历", 0, 2, "张三"))
        store.upsert_entity("张三", "人物", mention=Mention("d1", "病历", None, None, "张三"))

        assert store.get_entity("张三").traceable_ratio == 0.5

    def test_specific_type_not_overwritten_by_unknown(self):
        """后到的"未知"类型不能覆盖已经确定的类型。"""
        store = KnowledgeGraphStore()
        store.upsert_entity("呋塞米", "药物")
        store.upsert_entity("呋塞米", "未知")
        assert store.get_entity("呋塞米").entity_type == "药物"

    def test_invalid_name_returns_none(self):
        store = KnowledgeGraphStore()
        assert store.upsert_entity("的") is None
        assert len(store.entities) == 0


class TestRelationUpsert:
    def _store_with_two_entities(self):
        store = KnowledgeGraphStore()
        store.upsert_entity("张三", "患者")
        store.upsert_entity("扩张型心肌病", "诊断")
        return store

    def test_basic_relation(self):
        store = self._store_with_two_entities()
        ok = store.upsert_relation(
            GraphRelation("张三", "扩张型心肌病", "患有", evidence="既往诊断扩张型心肌病")
        )
        assert ok is True
        assert len(store.relations) == 1

    def test_self_loop_rejected(self):
        """自环对检索无价值，且会污染社区发现。"""
        store = self._store_with_two_entities()
        assert store.upsert_relation(GraphRelation("张三", "张三", "自身")) is False
        assert len(store.relations) == 0

    def test_dangling_endpoint_rejected(self):
        """端点不存在的边说明关系抽取幻觉了，必须拒绝。"""
        store = self._store_with_two_entities()
        assert store.upsert_relation(GraphRelation("张三", "不存在实体", "患有")) is False
        assert len(store.relations) == 0

    def test_duplicate_relation_boosts_confidence(self):
        """同一条边被多次抽到 -> 提高置信度，但封顶 1.0。"""
        store = self._store_with_two_entities()
        for _ in range(10):
            store.upsert_relation(
                GraphRelation("张三", "扩张型心肌病", "患有", evidence="证据", confidence=0.8)
            )
        assert len(store.relations) == 1, "同源同目标同类型应视为同一条边"
        rel = next(iter(store.relations.values()))
        assert rel.confidence == 1.0

    def test_adjacency_is_bidirectional(self):
        """多跳扩展使用无向视图：能从任一方向走通。"""
        store = self._store_with_two_entities()
        store.upsert_relation(GraphRelation("张三", "扩张型心肌病", "患有"))
        assert "扩张型心肌病" in store.neighbors("张三")
        assert "张三" in store.neighbors("扩张型心肌病")


# =============================================================================
# 从抽取结果建图
# =============================================================================


class TestAddExtractions:
    def test_counts_added_and_merged(self):
        store = KnowledgeGraphStore()
        extractions = [
            {"extraction_class": "疾病", "extraction_text": "扩张型心肌病",
             "attributes": {"性质": "心肌病变"}, "char_interval": {"start_pos": 10, "end_pos": 16}},
            {"extraction_class": "疾病", "extraction_text": "扩张型心肌病",
             "char_interval": {"start_pos": 300, "end_pos": 306}},
            {"extraction_class": "药物", "extraction_text": "呋塞米",
             "char_interval": {"start_pos": 400, "end_pos": 403}},
        ]
        result = store.add_extractions("d1", "病历.pdf", extractions)

        assert result["entities_added"] == 2
        assert result["entities_merged"] == 1
        assert len(store.entities) == 2

    def test_entity_class_filter(self):
        """可以只把指定类别当作实体，其余（如"关系描述"）跳过。"""
        store = KnowledgeGraphStore()
        extractions = [
            {"extraction_class": "疾病", "extraction_text": "心房颤动"},
            {"extraction_class": "关系描述", "extraction_text": "患者长期服用苯磺酸氨氯地平",
             "attributes": {"主体1": "张建国", "主体2": "苯磺酸氨氯地平", "关系": "服用"}},
        ]
        result = store.add_extractions("d1", "病历.pdf", extractions, entity_classes={"疾病"})
        assert result["entities_added"] == 1
        assert store.get_entity("心房颤动") is not None
        assert store.get_entity("患者长期服用苯磺酸氨氯地平") is None


class TestAddTriples:
    def test_relations_written_with_auto_created_entities(self):
        store = KnowledgeGraphStore()
        triples = [
            RelationTriple("张建国", "患有", "扩张型心肌病", "患者", "诊断", evidence="既往诊断"),
            RelationTriple("扩张型心肌病", "治疗", "沙库巴曲缬沙坦钠片", "诊断", "药物"),
        ]
        result = store.add_triples(triples, doc_id="d1")

        assert result["relations_added"] == 2
        assert len(store.entities) == 3
        assert store.stats()["relation_count"] == 2


# =============================================================================
# 实体链接
# =============================================================================


class TestFindEntities:
    @pytest.fixture
    def store(self):
        store = KnowledgeGraphStore()
        for name, etype in [
            ("张建国", "患者"), ("扩张型心肌病", "诊断"), ("沙库巴曲缬沙坦钠片", "药物"),
            ("北京协和医院", "机构"), ("北京", "地点"),
        ]:
            store.upsert_entity(name, etype)
        return store

    def test_exact_substring_match(self, store):
        hits = store.find_entities("张建国的诊断是什么")
        assert hits[0][0].name == "张建国"

    def test_longer_entity_preferred_over_shorter(self, store):
        """查询含"北京协和医院"时，不应优先返回更短的"北京"。"""
        hits = store.find_entities("患者在北京协和医院就诊")
        names = [e.name for e, _ in hits]
        assert "北京协和医院" in names
        assert names.index("北京协和医院") <= names.index("北京")

    def test_no_match_returns_empty(self, store):
        assert store.find_entities("完全无关的问题内容") == []

    def test_empty_query(self, store):
        assert store.find_entities("") == []

    def test_fuzzy_tolerates_typo(self, store):
        """错别字场景："扩张性心肌病"（性 vs 型）应能模糊命中。"""
        hits = store.find_entities("扩张性心肌病怎么治")
        assert any(e.name == "扩张型心肌病" for e, _ in hits)


# =============================================================================
# 多跳检索 —— 本文件最重要的部分
# =============================================================================


@pytest.fixture
def medical_graph():
    """构造一个医学知识图谱。

    刻意设计成"答案跨 3 个节点"：
        张建国 --患有--> 扩张型心肌病 --治疗--> 沙库巴曲缬沙坦钠片 --研发--> 诺华

    任何**单个文本块**都不可能同时包含"张建国"和"诺华"，
    所以纯向量检索答不出"张建国用的药是哪家公司研发的"。
    图检索通过 3 跳遍历可以。

    建图走的是真实链路的两步：
      1. `add_extractions` 建实体（带 char_interval，因此可溯源）
      2. `add_triples(auto_create_entities=False)` 只补关系边
    """
    store = KnowledgeGraphStore()

    entities = [
        ("患者", "张建国", 10), ("诊断", "扩张型心肌病", 40), ("药物", "沙库巴曲缬沙坦钠片", 80),
        ("机构", "诺华", 120), ("症状", "双下肢水肿", 160), ("机构", "北京协和医院", 200),
    ]
    store.add_extractions(
        "d1", "病历.pdf",
        [
            {
                "extraction_class": etype,
                "extraction_text": name,
                "attributes": {},
                "char_interval": {"start_pos": pos, "end_pos": pos + len(name)},
            }
            for etype, name, pos in entities
        ],
    )

    triples = [
        RelationTriple("张建国", "患有", "扩张型心肌病", evidence="诊断：扩张型心肌病"),
        RelationTriple("扩张型心肌病", "治疗", "沙库巴曲缬沙坦钠片", evidence="沙库巴曲缬沙坦钠片 50mg po bid"),
        RelationTriple("沙库巴曲缬沙坦钠片", "研发", "诺华", evidence="诺华研发的沙库巴曲缬沙坦钠"),
        RelationTriple("扩张型心肌病", "表现", "双下肢水肿", evidence="双下肢凹陷性水肿"),
        RelationTriple("张建国", "就诊于", "北京协和医院", evidence="就诊于北京协和医院"),
    ]
    store.add_triples(triples, doc_id="d1", doc_title="病历.pdf", auto_create_entities=False)
    return store


class TestMultiHopReachability:
    def test_one_hop(self, medical_graph):
        assert medical_graph.neighbors("张建国", hops=1) == {"扩张型心肌病", "北京协和医院"}

    def test_three_hops_reaches_company(self, medical_graph):
        """核心断言：3 跳内必须能到达"诺华"。"""
        reached = medical_graph.neighbors("张建国", hops=3)
        assert "诺华" in reached

    def test_one_hop_does_not_reach_company(self, medical_graph):
        """对照组：1 跳够不到。这解释了为什么"多跳"是必要的，而不是炫技。"""
        assert "诺华" not in medical_graph.neighbors("张建国", hops=1)

    def test_subgraph_contains_cross_chunk_edge(self, medical_graph):
        seeds = [e.entity_id for e, _ in medical_graph.find_entities("张建国")]
        sub = medical_graph.subgraph(seeds, hops=3)

        node_set = set(sub["nodes"])
        assert "诺华" in node_set
        assert normalize_entity_name("沙库巴曲缬沙坦钠片") in node_set
        # 子图内的边必须两端都在节点集合内，否则会给出悬空引用
        for edge in sub["edges"]:
            assert edge.source_id in node_set and edge.target_id in node_set

    def test_subgraph_respects_node_budget(self):
        """节点预算必须生效，否则热门实体会把 LLM 上下文撑爆。"""
        store = KnowledgeGraphStore()
        store.add_triples([RelationTriple("中心", "连接", f"叶子{i}") for i in range(50)])
        sub = store.subgraph(["中心"], hops=1, max_nodes=10)

        assert len(sub["nodes"]) <= 10
        assert sub["truncated"] is True

    def test_seeds_always_included(self):
        store = KnowledgeGraphStore()
        store.upsert_entity("孤立实体", "其他")
        sub = store.subgraph(["孤立实体"], hops=2)
        assert sub["nodes"] == ["孤立实体"]

    def test_empty_seeds(self, medical_graph):
        sub = medical_graph.subgraph([], hops=2)
        assert sub["nodes"] == [] and sub["edges"] == []


class TestRelationChains:
    def test_chain_is_human_readable(self, medical_graph):
        chains = medical_graph.relation_chains(["张建国"], max_hops=3, max_chains=20)
        texts = [c["text"] for c in chains]
        # 必须存在一条到达诺华的关系链，且带箭头可读
        assert any("诺华" in t and "--[" in t for t in texts), texts

    def test_chains_respect_max(self, medical_graph):
        chains = medical_graph.relation_chains(["张建国"], max_hops=3, max_chains=3)
        assert len(chains) <= 3

    def test_chain_has_no_repeated_entity(self, medical_graph):
        """简单路径：链内不能重复实体，否则会出现 A->B->A 这种无信息量的环。"""
        for chain in medical_graph.relation_chains(["张建国"], max_hops=3, max_chains=20):
            ids = [node["entity_id"] for node in chain["chain"]]
            assert len(ids) == len(set(ids))

    def test_hop_count_matches_relation_count(self, medical_graph):
        for chain in medical_graph.relation_chains(["张建国"], max_hops=3):
            assert chain["hops"] == len(chain["relations"])
            assert chain["hops"] == len(chain["chain"]) - 1


class TestShortestPath:
    def test_path_between_cross_chunk_entities(self, medical_graph):
        path = medical_graph.shortest_path("张建国", "诺华", max_hops=4)
        assert path is not None
        names = [node["name"] for node in path["chain"]]
        assert names[0] == "张建国" and names[-1] == "诺华"
        assert path["hops"] == 3

    def test_direct_relation_is_one_hop(self, medical_graph):
        path = medical_graph.shortest_path("张建国", "扩张型心肌病")
        assert path["hops"] == 1

    def test_unknown_entity_returns_none(self, medical_graph):
        assert medical_graph.shortest_path("张建国", "不存在的公司") is None

    def test_max_hops_limit(self, medical_graph):
        """跳数上限生效时返回 None 而不是绕远路。"""
        assert medical_graph.shortest_path("双下肢水肿", "诺华", max_hops=1) is None


# =============================================================================
# 社区发现（GraphRAG global search 的基础）
# =============================================================================


class TestCommunities:
    def test_two_disjoint_clusters_detected(self):
        """两个互不相连的簇必须被分成两个社区。"""
        store = KnowledgeGraphStore()
        store.add_triples([
            RelationTriple("A1", "相关", "A2"),
            RelationTriple("A2", "相关", "A3"),
            RelationTriple("B1", "相关", "B2"),
            RelationTriple("B2", "相关", "B3"),
        ])
        communities = store.detect_communities()
        sizes = sorted(len(c) for c in communities)
        assert sizes == [3, 3]

    def test_communities_cover_all_nodes(self):
        store = KnowledgeGraphStore()
        store.add_triples([
            RelationTriple("A", "相关", "B"),
            RelationTriple("C", "相关", "D"),
            RelationTriple("孤立", "相关", "孤立2"),
        ])
        covered = set()
        for comm in store.detect_communities():
            covered |= comm
        assert covered == set(store.entities.keys())

    def test_deterministic(self, medical_graph):
        """同一张图跑两次必须得到相同划分（否则指标无法复现）。"""
        first = [sorted(c) for c in medical_graph.detect_communities()]
        second = [sorted(c) for c in medical_graph.detect_communities()]
        assert first == second

    def test_modularity_of_clustered_graph_is_positive(self):
        store = KnowledgeGraphStore()
        store.add_triples([
            RelationTriple("A1", "相关", "A2"),
            RelationTriple("A2", "相关", "A3"),
            RelationTriple("A3", "相关", "A1"),  # 簇内成环
            RelationTriple("B1", "相关", "B2"),
            RelationTriple("B2", "相关", "B3"),
            RelationTriple("B3", "相关", "B1"),
            RelationTriple("A1", "相关", "B1"),   # 唯一的桥
        ])
        q = store.modularity(store.detect_communities())
        assert q > 0.2, f"存在明显社区结构时模块度应显著为正，实际 Q={q}"

    def test_modularity_zero_for_empty_graph(self):
        assert KnowledgeGraphStore().modularity([]) == 0.0


# =============================================================================
# 统计与持久化
# =============================================================================


class TestStats:
    def test_stats_fields(self, medical_graph):
        stats = medical_graph.stats()
        assert stats["entity_count"] == 6
        assert stats["relation_count"] == 5
        assert stats["document_count"] == 1
        assert stats["traceable_ratio"] == 1.0
        assert stats["mention_count"] == stats["entity_count"]
        assert "诊断" in stats["entity_type_distribution"]

    def test_avg_degree(self, medical_graph):
        # 5 条边、6 个节点 -> 2*5/6 ≈ 1.667
        assert medical_graph.stats()["avg_degree"] == pytest.approx(1.667, abs=0.01)

    def test_empty_graph_stats(self):
        stats = KnowledgeGraphStore().stats()
        assert stats["entity_count"] == 0
        assert stats["traceable_ratio"] == 0.0
        assert stats["avg_degree"] == 0.0


class TestPersistence:
    def test_roundtrip_preserves_everything(self, medical_graph, workspace_tmp):
        path = os.path.join(workspace_tmp, "graph.json")
        medical_graph.save(path)
        reloaded = KnowledgeGraphStore.load(path)

        assert reloaded.stats() == medical_graph.stats()
        assert reloaded.shortest_path("张建国", "诺华")["hops"] == 3
        assert reloaded.get_entity("张建国").traceable_ratio == 1.0
        # 边类型索引必须一起恢复，否则关系链会退化成"相关"
        assert reloaded._relation_type_between("张建国", "扩张型心肌病") == "患有"

    def test_saved_file_is_valid_utf8_json(self, medical_graph, workspace_tmp):
        """中文不能变成 \\uXXXX 转义——否则人没法直接看图谱文件。"""
        path = os.path.join(workspace_tmp, "graph.json")
        medical_graph.save(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        assert "扩张型心肌病" in raw
        assert json.loads(raw)["version"] == 1

    def test_load_missing_file_returns_empty_graph(self, workspace_tmp):
        store = KnowledgeGraphStore.load(os.path.join(workspace_tmp, "definitely_missing.json"))
        assert store.stats()["entity_count"] == 0

    def test_clear(self, medical_graph):
        medical_graph.clear()
        assert medical_graph.stats()["entity_count"] == 0
        assert medical_graph.stats()["relation_count"] == 0
        assert medical_graph.neighbors("张建国") == set()
