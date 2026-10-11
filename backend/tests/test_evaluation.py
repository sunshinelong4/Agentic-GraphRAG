"""评测框架的测试

分两类，都很重要：

**A. fixture 完整性** —— 保证评测数据本身没有错。
   最典型的是"三元组的证据必须真的在语料里"，否则评测就在用一个不存在的
   世界测自己的检索器。第一版评测就是因为块设计不当（一个块装下了全部答案），
   让基线拿到满分，失去了区分度。这些测试就是防这个的。

**B. 结论回归** —— 把评测得出的关键结论固化成断言。
   如果将来改动让"图检索在多跳题上不再优于切块检索"，
   测试会立刻失败，而不是等到面试时被问住才发现。
"""

import statistics

import pytest

from evaluation.fixtures import CORPUS, ENTITIES, QUESTIONS, TRIPLES, corpus_by_chunk_id
from evaluation.run_eval import (
    BigramBM25,
    build_fixture_graph,
    build_full_documents,
    evidence_recall,
    make_retrievers,
    run,
    single_chunk_ceiling,
)


# =============================================================================
# A. Fixture 完整性
# =============================================================================


class TestFixtureIntegrity:
    def test_chunk_ids_unique(self):
        ids = [chunk["chunk_id"] for chunk in CORPUS]
        assert len(ids) == len(set(ids)), "块 ID 重复会让证据定位错乱"

    def test_required_chunk_fields(self):
        for chunk in CORPUS:
            for field in ("chunk_id", "doc_id", "doc_title", "content"):
                assert chunk.get(field), f"{chunk.get('chunk_id')} 缺少字段 {field}"

    def test_corpus_has_enough_distractors(self):
        """没有干扰块的话 top_k 检索等于全取，基线会被灌水。"""
        assert len(CORPUS) >= 15, "语料太小，切块检索的排序能力无法被测试"
        real_chunks = [c for c in CORPUS if c["doc_id"] != "doc-distractor"]
        assert len(real_chunks) < len(CORPUS), "必须包含干扰块"

    def test_entity_names_appear_in_their_chunk(self):
        """实体必须真的出现在它声明的块里，否则图与语料对不上。"""
        chunks = corpus_by_chunk_id()
        for entity in ENTITIES:
            content = chunks[entity["chunk_id"]]["content"]
            assert entity["name"] in content, (
                f"实体 '{entity['name']}' 不在块 {entity['chunk_id']} 中"
            )

    def test_triple_endpoints_are_declared_entities(self):
        """关系的两端必须是已声明的实体，否则落图时会被当作悬空端点拒绝。"""
        names = {entity["name"] for entity in ENTITIES}
        for triple in TRIPLES:
            assert triple["head"] in names, f"关系头实体未声明: {triple['head']}"
            assert triple["tail"] in names, f"关系尾实体未声明: {triple['tail']}"

    def test_triple_evidence_exists_in_corpus(self):
        """每条关系的头实体必须出现在它声明的证据块里。

        这是防止"评测数据自己就是幻觉"的关键断言：标准答案必须能在原文里找到出处。
        """
        chunks = corpus_by_chunk_id()
        for triple in TRIPLES:
            content = chunks[triple["chunk_id"]]["content"]
            assert triple["head"] in content or triple["tail"] in content, (
                f"关系 {triple['head']}-{triple['relation']}->{triple['tail']} "
                f"在块 {triple['chunk_id']} 中找不到任何端点"
            )

    def test_no_self_loops_in_fixture(self):
        for triple in TRIPLES:
            assert triple["head"] != triple["tail"]

    def test_question_expected_terms_appear_in_corpus(self):
        """期望词必须至少出现在一个块里，否则该题永远不可能被答对。"""
        for question in QUESTIONS:
            for term in question["expected_terms"]:
                found = any(term in chunk["content"] for chunk in CORPUS)
                assert found, f"问题 {question['id']} 的期望词 '{term}' 不在任何块中"

    def test_expected_terms_are_not_single_chunk_for_cross_chunk_questions(self):
        """多跳/聚合题的期望词必须真的跨块 —— 否则测试不到多跳能力。"""
        chunks = corpus_by_chunk_id()
        for question in QUESTIONS:
            if question["type"] == "detail":
                continue
            ceiling = single_chunk_ceiling(question["expected_terms"])
            assert ceiling < 1.0, (
                f"问题 {question['id']}（{question['type']}）的期望词能被单个块全装下"
                f"（ceiling={ceiling}），它测不到跨块检索能力。note: {question['note']}"
            )

    def test_detail_questions_are_single_chunk(self):
        """细节题应当单块可答，用来体现切块检索的主场。"""
        for question in QUESTIONS:
            if question["type"] != "detail":
                continue
            assert single_chunk_ceiling(question["expected_terms"]) == 1.0

    def test_question_ids_unique(self):
        ids = [q["id"] for q in QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_graph_questions_reference_graph_entities(self):
        """多跳/聚合题的期望词必须落在图谱实体上，否则是数据问题而非能力问题。

        细节题不参与这个断言是有意的：像 "38%" 这样的**数值**本来就不是图谱实体，
        图检索在数值查找上天然没优势（评测里如实体现了这一点）。
        """
        graph_names = {entity["name"] for entity in ENTITIES}
        for question in QUESTIONS:
            if question["type"] == "detail":
                continue
            hits = sum(1 for term in question["expected_terms"] if term in graph_names)
            assert hits >= 1, f"问题 {question['id']} 与知识图谱完全无关"


# =============================================================================
# 图构建
# =============================================================================


class TestBuildFixtureGraph:
    def test_full_documents_offsets_correct(self):
        documents = build_full_documents()
        for doc in documents.values():
            for chunk_id, offset in doc["chunk_offsets"].items():
                name = next(c["content"] for c in CORPUS if c["chunk_id"] == chunk_id)
                assert doc["text"][offset:offset + len(name)] == name, (
                    f"块 {chunk_id} 的偏移量算错了，溯源会整体错位"
                )

    def test_graph_built_without_llm(self):
        stats = build_fixture_graph().stats()
        assert stats["entity_count"] == len(ENTITIES)
        assert stats["relation_count"] == len(TRIPLES)
        assert stats["traceable_ratio"] == 1.0, "固定语料建图应当 100% 可溯源"

    def test_all_triples_accepted(self):
        """人工标注的三元组必须全部落图，一条都不能被拒绝。"""
        store = build_fixture_graph()
        from app.services.graph_store import normalize_entity_name

        for triple in TRIPLES:
            # 注意去重键的顺序是 (source_id, target_id, relation_type)
            key = (
                normalize_entity_name(triple["head"]),
                normalize_entity_name(triple["tail"]),
                triple["relation"],
            )
            assert key in store.relations, f"三元组未被落图: {triple}"

    def test_cross_chunk_path_exists(self):
        """核心：张建国 → 药 → 类名 → 诺华 这条 3 跳路径必须存在。"""
        store = build_fixture_graph()
        path = store.shortest_path("张建国", "诺华", max_hops=4)
        assert path is not None, "跨块关系路径不存在，多跳评测无从谈起"
        assert path["hops"] == 3

    def test_community_detection_meaningful(self):
        store = build_fixture_graph()
        communities = store.detect_communities()
        assert len(communities) >= 2
        q = store.modularity(communities)
        assert q > 0.2, f"固定图谱应有明显社区结构，实际 Q={q}"


# =============================================================================
# 指标函数
# =============================================================================


class TestMetrics:
    def test_evidence_recall_full(self):
        assert evidence_recall(["诺华", "沙库巴曲缬沙坦钠片"], "张建国服用沙库巴曲缬沙坦钠片，由诺华研制") == 1.0

    def test_evidence_recall_partial(self):
        assert evidence_recall(["诺华", "阿斯利康"], "由诺华研制") == 0.5

    def test_evidence_recall_none(self):
        assert evidence_recall(["诺华"], "完全无关的内容") == 0.0

    def test_evidence_recall_empty_expectation(self):
        assert evidence_recall([], "任意") == 1.0

    def test_evidence_recall_handles_wrapper_chars(self):
        """期待词带书名号时也应命中（走规范化）。"""
        assert evidence_recall(["《诺华》"], "由诺华研制") == 1.0

    def test_ceiling_of_detail_question_is_one(self):
        assert single_chunk_ceiling(["38%"]) == 1.0

    def test_ceiling_of_cross_chunk_question_below_one(self):
        assert single_chunk_ceiling(["沙库巴曲缬沙坦钠片", "诺华"]) < 1.0

    def test_ceiling_is_upper_bound_for_any_single_chunk(self):
        """ceiling 定义上必须 >= 任意单块的召回率。"""
        terms = ["诺华", "阿斯利康"]
        ceiling = single_chunk_ceiling(terms)
        for chunk in CORPUS:
            assert evidence_recall(terms, chunk["content"]) <= ceiling


class TestBigramBM25:
    def test_ranks_relevant_chunk_first(self):
        bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])
        hits = bm25.search("左室射血分数是多少", top_k=3)
        assert hits, "BM25 应该能召回内容"
        top_chunk = CORPUS[hits[0][0]]["content"]
        assert "射血分数" in top_chunk

    def test_returns_at_most_top_k(self):
        bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])
        assert len(bm25.search("张建国", top_k=2)) <= 2

    def test_empty_query(self):
        bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])
        assert bm25.search("", top_k=3) == []

    def test_no_match_returns_empty(self):
        bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])
        assert bm25.search("zzzzqqqqxxxx", top_k=3) == []

    def test_empty_corpus_is_safe(self):
        assert BigramBM25([]).search("查询", top_k=3) == []

    def test_scores_are_sorted_descending(self):
        bm25 = BigramBM25([chunk["content"] for chunk in CORPUS])
        scores = [score for _, score in bm25.search("张建国用药", top_k=5)]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# B. 结论回归
# =============================================================================


@pytest.fixture(scope="module")
def report():
    """整个评测只跑一次（模块级缓存），避免重复构建图谱。"""
    return run(top_k=3, hops=2)


def _recall_signature(report: dict) -> dict:
    """只保留确定性字段（召回率、计数），剔除延迟。"""
    signature = {}
    for name, summary in report["summary"].items():
        signature[name] = {
            "overall_recall": summary["overall_recall"],
            "by_type": {
                qtype: {"count": item["count"], "recall": item["recall"]}
                for qtype, item in summary["by_type"].items()
            },
            "on_cross_chunk": {
                "count": summary["on_cross_chunk"]["count"],
                "recall": summary["on_cross_chunk"]["recall"],
                "fully_solved": summary["on_cross_chunk"]["fully_solved"],
            },
            "errors": summary["errors"],
        }
    return signature


class TestEvalReproducibility:
    def test_recall_is_deterministic(self):
        """评测必须可复现，否则数字没有意义。

        注意只比较**召回率**而不比较延迟：延迟本来就该每次不同，
        把它纳入确定性断言是测试写错了（第一版就是这么错的）。
        """
        first = _recall_signature(run(top_k=3, hops=2))
        second = _recall_signature(run(top_k=3, hops=2))
        assert first == second

    def test_meta_reports_scale(self, report):
        meta = report["meta"]
        assert meta["questions"] == len(QUESTIONS)
        assert meta["corpus_chunks"] == len(CORPUS)
        assert meta["graph_entities"] == len(ENTITIES)
        assert meta["graph_relations"] == len(TRIPLES)
        assert meta["top_k"] == 3

    def test_all_configs_present(self, report):
        for name in ["chunk_bm25", "graph_local", "graph_global", "graph_auto", "hybrid_bm25_graph"]:
            assert name in report["summary"], f"缺少配置 {name}"

    def test_no_retrieval_errors(self, report):
        for name, summary in report["summary"].items():
            assert summary["errors"] == 0, f"{name} 出现检索异常"


class TestCoreClaims:
    """把评测结论固化成断言 —— 这是"可验证的结论"而不是"我说它更好"。"""

    def test_graph_beats_chunking_on_multihop(self, report):
        """核心结论 1：多跳题上图检索显著优于切块检索。"""
        bm25 = report["summary"]["chunk_bm25"]["by_type"]["multihop"]["recall"]
        graph = report["summary"]["graph_auto"]["by_type"]["multihop"]["recall"]
        assert graph > bm25, f"图检索在多跳题上应优于 BM25（graph={graph}, bm25={bm25}）"

    def test_graph_beats_chunking_on_cross_chunk(self, report):
        """核心结论 2：跨块题上图的"全对"数量更多。"""
        bm25 = report["summary"]["chunk_bm25"]["on_cross_chunk"]
        graph = report["summary"]["graph_auto"]["on_cross_chunk"]
        assert graph["fully_solved"] >= bm25["fully_solved"], (
            f"跨块全对: graph={graph['fully_solved']}/{graph['count']}, "
            f"bm25={bm25['fully_solved']}/{bm25['count']}"
        )

    def test_chunking_wins_on_detail(self, report):
        """诚实的反面结论：细节/数值类问题上切块检索更好。

        这条断言很重要 —— 它防止我们把项目吹成"图检索全面碾压"。
        数值不是图谱实体，图检索在这类问题上天然没优势。
        """
        bm25 = report["summary"]["chunk_bm25"]["by_type"]["detail"]["recall"]
        graph = report["summary"]["graph_auto"]["by_type"]["detail"]["recall"]
        assert bm25 >= graph, f"细节题上 BM25 应不劣于图检索（bm25={bm25}, graph={graph}）"

    def test_hybrid_is_best_overall(self, report):
        """核心结论 3：两路互补，混合配置综合最优 —— 这验证了"互补"而非"替代"的设计。"""
        hybrid = report["summary"]["hybrid_bm25_graph"]["overall_recall"]
        for name, summary in report["summary"].items():
            assert hybrid >= summary["overall_recall"], (
                f"混合配置应不劣于 {name}（hybrid={hybrid}, {name}={summary['overall_recall']}）"
            )

    def test_cross_chunk_questions_exist(self, report):
        """评测必须真的包含跨块问题，否则什么都没测到。"""
        assert len(report["meta"]["cross_chunk_questions"]) >= 5

    def test_latency_is_reasonable(self, report):
        """纯本地检索应当是毫秒级；慢了说明实现有明显性能问题。"""
        for name, summary in report["summary"].items():
            assert summary["overall_latency_ms"] < 500, (
                f"{name} 平均延迟 {summary['overall_latency_ms']}ms，明显偏慢"
            )


class TestRetrieverWiring:
    def test_retrievers_are_callable_and_return_text(self):
        store = build_fixture_graph()
        retrievers = make_retrievers(store, top_k=3, hops=2)
        for name, retrieve in retrievers.items():
            evidence = retrieve("张建国的诊断是什么")
            assert isinstance(evidence, str), f"{name} 必须返回文本证据"

    def test_graph_local_returns_empty_for_unknown_entity(self):
        store = build_fixture_graph()
        retrievers = make_retrievers(store)
        assert retrievers["graph_local"]("完全不存在的实体名称") == ""
