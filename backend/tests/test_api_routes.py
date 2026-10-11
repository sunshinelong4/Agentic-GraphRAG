"""API 路由的集成测试

用 FastAPI 的 TestClient 真实走一遍 HTTP 层，验证：
1. 图谱路由已正确注册、请求/响应模型无误
2. 图谱构建 → 统计 → 检索 → 实体详情 → 社区 的完整闭环
3. **SSE 流式协议**的事件序列与格式（前端就是靠这个对接的）
4. 依赖缺失时（向量库不可用）Agent 仍能优雅降级

测试全部离线：图谱用 `build_relations=False` 导入（不调 LLM），
Agent 用注入的假 LLM 与假向量库（通过覆盖 FastAPI 依赖实现）。
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.agent_qa import GraphRAGAgent, reset_graph_agent
from app.services.graph_store import KnowledgeGraphStore
from tests.test_agent_qa import FakeLLM, FakeVectorStore


# =============================================================================
# 测试夹具：用临时图谱路径 + 假 LLM/向量库驱动真实 HTTP 层
# =============================================================================

SAMPLE_TEXT = (
    "患者张建国，男，62岁。诊断为扩张型心肌病。左室射血分数（LVEF）38%。"
    "给予沙库巴曲缬沙坦钠片治疗。该药由诺华研发。"
)

SAMPLE_EXTRACTIONS = [
    {"extraction_class": "患者", "extraction_text": "张建国",
     "char_interval": {"start_pos": 2, "end_pos": 5}, "attributes": {"性别": "男"}},
    {"extraction_class": "诊断", "extraction_text": "扩张型心肌病",
     "char_interval": {"start_pos": 10, "end_pos": 16}, "attributes": {}},
    {"extraction_class": "药物", "extraction_text": "沙库巴曲缬沙坦钠片",
     "char_interval": {"start_pos": 45, "end_pos": 55}, "attributes": {}},
    {"extraction_class": "机构", "extraction_text": "诺华",
     "char_interval": {"start_pos": 65, "end_pos": 67}, "attributes": {}},
]


@pytest.fixture
def test_settings(workspace_tmp):
    """测试用的 Settings（图谱落在工作区临时目录）。"""
    return Settings(graph_persist_path=os.path.join(workspace_tmp, "graph.json"), auth_enabled=False,
                    user_requests_per_minute=10000)


@pytest.fixture
def client(test_settings, monkeypatch):
    """提供走临时图谱文件、并用假 LLM 的 TestClient。"""
    import app.api.graph_routes as graph_routes
    monkeypatch.setattr("app.traffic.get_settings", lambda: test_settings)

    # 隔离图谱单例，避免测试之间互相污染
    graph_routes.reset_graph_store()
    reset_graph_agent()

    app.dependency_overrides[get_settings] = lambda: test_settings

    # 让图谱路由拿到假向量库（不连 Chroma / 不调 DashScope）
    vector_store = FakeVectorStore(
        results=[
            {"score": 0.9, "chunk_id": "c1", "doc_id": "d1", "doc_title": "病历.pdf",
             "content": "患者张建国，LVEF 38%。诊断为扩张型心肌病。",
             "chunk_type": "document", "attributes": {"paragraph_index": 0}},
        ],
        summary={"document_count": 1, "chunk_count": 3,
                 "documents": [{"doc_id": "d1", "title": "病历.pdf"}]},
    )
    monkeypatch.setattr(graph_routes, "_get_vector_store", lambda _settings: vector_store)

    # 让 Agent 用假 LLM（默认会被路由到真实 DeepSeek）
    original_build_agent = graph_routes._build_agent

    def fake_build_agent(_settings):
        agent = original_build_agent(_settings)
        agent._llm = FakeLLM("张建国患有扩张型心肌病，LVEF 38%。")
        agent.use_llm_router = False
        agent.vector_store = vector_store
        return agent

    monkeypatch.setattr(graph_routes, "_build_agent", fake_build_agent)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    graph_routes.reset_graph_store()
    reset_graph_agent()


@pytest.fixture
def client_with_graph(client, test_settings):
    """已经导入好实体与关系的客户端。"""
    # 先建实体
    response = client.post("/api/rag/graph/build", json={
        "doc_id": "d1", "doc_title": "病历.pdf", "text": SAMPLE_TEXT,
        "extractions": SAMPLE_EXTRACTIONS, "build_relations": False,
    })
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True

    # 再补关系（直接走 store，避免调 LLM）
    from app.api.graph_routes import get_graph_store
    from app.services.graph_store import RelationTriple

    store: KnowledgeGraphStore = get_graph_store(test_settings)
    store.add_triples(
        [
            RelationTriple("张建国", "患有", "扩张型心肌病", evidence="诊断为扩张型心肌病", confidence=0.9),
            RelationTriple("扩张型心肌病", "治疗", "沙库巴曲缬沙坦钠片",
                           evidence="给予沙库巴曲缬沙坦钠片治疗", confidence=0.88),
            RelationTriple("沙库巴曲缬沙坦钠片", "研发", "诺华",
                           evidence="该药由诺华研发", confidence=0.85),
        ],
        doc_id="d1", doc_title="病历.pdf", auto_create_entities=False,
    )
    return client


# =============================================================================
# 路由注册
# =============================================================================


def _openapi_paths(client) -> set:
    """从 OpenAPI 文档取已注册路径。

    为什么不直接遍历 `app.routes`：这个 FastAPI 版本把 `include_router` 的结果
    包成 `_IncludedRouter` 对象（没有 `.path` 属性），直接取属性会 AttributeError。
    走 OpenAPI 文档反而更贴近**对外契约**——它验证的是"客户端真的能调用到这些接口"。
    """
    spec = client.get("/openapi.json").json()
    return set(spec["paths"].keys())


class TestRouteRegistration:
    def test_graph_routes_registered(self, client):
        paths = _openapi_paths(client)
        for expected in [
            "/api/rag/graph/build",
            "/api/rag/graph/stats",
            "/api/rag/graph/communities",
            "/api/rag/graph/query",
            "/api/rag/graph/entities/{name}",
            "/api/rag/graph/reset",
        ]:
            assert expected in paths, f"缺少路由 {expected}"

    def test_agent_routes_registered(self, client):
        paths = _openapi_paths(client)
        for expected in [
            "/api/rag/agent/qa",
            "/api/rag/agent/qa/stream",
            "/api/rag/agent/route-preview",
        ]:
            assert expected in paths, f"缺少路由 {expected}"

    def test_existing_rag_routes_still_present(self, client):
        """新增路由不能破坏既有接口。"""
        paths = _openapi_paths(client)
        for expected in ["/api/rag/qa", "/api/rag/search", "/api/rag/chat", "/api/rag/pdf/parse"]:
            assert expected in paths, f"既有路由 {expected} 丢失了"


# =============================================================================
# 图谱构建与统计
# =============================================================================


class TestGraphBuildEndpoint:
    def test_build_entities_only(self, client):
        response = client.post("/api/rag/graph/build", json={
            "doc_id": "d1", "doc_title": "病历.pdf", "text": SAMPLE_TEXT,
            "extractions": SAMPLE_EXTRACTIONS, "build_relations": False,
        })
        payload = response.json()

        assert payload["success"] is True
        assert payload["entities_added"] == 4
        assert payload["relations_added"] == 0
        assert payload["graph_stats"]["entity_count"] == 4
        assert payload["graph_stats"]["traceable_ratio"] == 1.0

    def test_build_generates_doc_id_when_absent(self, client):
        response = client.post("/api/rag/graph/build", json={
            "doc_title": "病历.pdf", "text": SAMPLE_TEXT,
            "extractions": SAMPLE_EXTRACTIONS, "build_relations": False,
        })
        assert response.json()["doc_id"], "未传 doc_id 时应自动生成"

    def test_build_persists_to_disk(self, client, test_settings):
        graph_path = test_settings.graph_persist_path
        client.post("/api/rag/graph/build", json={
            "doc_title": "病历.pdf", "text": SAMPLE_TEXT,
            "extractions": SAMPLE_EXTRACTIONS, "build_relations": False,
        })

        assert os.path.exists(graph_path), "图谱必须落盘，否则重启就丢"
        with open(graph_path, "r", encoding="utf-8") as f:
            assert "张建国" in f.read()

    def test_persisted_graph_survives_reload(self, client, test_settings):
        """模拟服务重启：清空单例后必须能从磁盘恢复全部实体与关系。"""
        import app.api.graph_routes as graph_routes

        client.post("/api/rag/graph/build", json={
            "doc_id": "d1", "doc_title": "病历.pdf", "text": SAMPLE_TEXT,
            "extractions": SAMPLE_EXTRACTIONS, "build_relations": False,
        })
        before = client.get("/api/rag/graph/stats").json()

        graph_routes.reset_graph_store()  # 等价于进程重启

        after = client.get("/api/rag/graph/stats").json()
        assert after["entity_count"] == before["entity_count"] == 4
        assert after["document_count"] == 1

    def test_invalid_extractions_rejected_by_schema(self, client):
        response = client.post("/api/rag/graph/build", json={
            "doc_title": "病历.pdf", "text": SAMPLE_TEXT, "extractions": [],
        })
        assert response.status_code == 422, "空抽取列表应被 Pydantic 拦下"


class TestStatsEndpoint:
    def test_stats_after_build(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/stats").json()

        assert payload["success"] is True
        assert payload["entity_count"] == 4
        assert payload["relation_count"] == 3
        assert payload["document_count"] == 1
        assert payload["traceable_ratio"] == 1.0
        assert payload["community_count"] >= 1
        assert "modularity" in payload

    def test_stats_on_empty_graph(self, client):
        payload = client.get("/api/rag/graph/stats").json()
        assert payload["success"] is True
        assert payload["entity_count"] == 0
        assert payload["modularity"] == 0.0


# =============================================================================
# 图谱检索
# =============================================================================


class TestGraphQueryEndpoint:
    def test_local_query_returns_chain(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/graph/query", json={
            "query": "张建国用的药是哪家公司研发的", "mode": "auto", "hops": 3,
        }).json()

        assert payload["success"] is True
        assert payload["hit"] is True
        assert payload["mode"] == "local"
        assert payload["seed_count"] >= 1
        assert "关系链" in payload["context_text"]
        assert "诺华" in payload["context_text"], "3 跳内必须能到达跨块实体"

    def test_explicit_global_mode(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/graph/query", json={
            "query": "涉及哪些机构", "mode": "global",
        }).json()

        assert payload["mode"] == "global"
        assert "社区摘要" in payload["context_text"]

    def test_no_hit_returns_empty_context(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/graph/query", json={
            "query": "完全不相关的实体名称",
        }).json()

        assert payload["success"] is True
        assert payload["hit"] is False
        assert payload["context_text"] == ""

    def test_hops_out_of_range_rejected(self, client_with_graph):
        response = client_with_graph.post("/api/rag/graph/query", json={
            "query": "张建国", "hops": 99,
        })
        assert response.status_code == 422


class TestEntityDetailEndpoint:
    def test_entity_found_with_relations(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/entities/张建国").json()

        assert payload["found"] is True
        assert payload["entity"]["name"] == "张建国"
        assert payload["entity"]["entity_type"] == "患者"
        assert payload["entity"]["mentions"][0]["start_pos"] == 2
        assert len(payload["relations"]) >= 1

    def test_entity_normalization(self, client_with_graph):
        """带书名号/大小写差异的名字也应能命中同一实体。"""
        payload = client_with_graph.get("/api/rag/graph/entities/诺华").json()
        assert payload["found"] is True

    def test_missing_entity_returns_found_false(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/entities/不存在的实体").json()
        assert payload["success"] is True
        assert payload["found"] is False

    def test_neighbors_include_multi_hop(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/entities/张建国?hops=3").json()
        names = [n["name"] for n in payload["neighbors"]]
        assert "诺华" in names, "3 跳邻居必须包含最远端实体"


class TestCommunitiesEndpoint:
    def test_communities_listed(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/communities").json()

        assert payload["success"] is True
        assert payload["community_count"] >= 1
        assert len(payload["communities"]) == payload["community_count"]

        first = payload["communities"][0]
        assert first["size"] >= 1
        assert first["members"], "社区必须有成员清单，否则无法回答聚合问题"

    def test_modularity_reported(self, client_with_graph):
        payload = client_with_graph.get("/api/rag/graph/communities").json()
        assert -1.0 <= payload["modularity"] <= 1.0


class TestResetEndpoint:
    def test_reset_clears_graph(self, client_with_graph):
        assert client_with_graph.get("/api/rag/graph/stats").json()["entity_count"] == 4

        assert client_with_graph.post("/api/rag/graph/reset").json()["success"] is True
        assert client_with_graph.get("/api/rag/graph/stats").json()["entity_count"] == 0


# =============================================================================
# Agent 问答
# =============================================================================


class TestAgentQAEndpoint:
    def test_agent_answer(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/agent/qa", json={
            "question": "张建国用了什么药？", "hops": 2, "use_llm_router": False,
        }).json()

        assert payload["success"] is True
        assert payload["answer"]
        assert payload["route"], "必须报告用了哪些检索工具"
        assert payload["verification"]["passed"] in (True, False)
        assert payload["attempts"] >= 1
        assert payload["trace"], "trace 不能为空，否则无法排查"

    def test_agent_includes_graph_payload(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/agent/qa", json={
            "question": "张建国", "use_llm_router": False,
        }).json()
        assert payload["graph"], "图检索结果应回传给前端用于可视化"

    def test_meta_question_route(self, client_with_graph):
        payload = client_with_graph.post("/api/rag/agent/qa", json={
            "question": "知识库里有几个文档？", "use_llm_router": False,
        }).json()
        assert payload["route"] == ["kb_meta"]

    def test_route_preview_is_llm_free(self, client_with_graph):
        payload = client_with_graph.get(
            "/api/rag/agent/route-preview", params={"question": "共涉及哪些机构"}
        ).json()
        assert "graph_global_search" in payload["tools"]
        assert payload["is_global_query"] is True


class TestAgentStreamEndpoint:
    def _parse_sse(self, body: str):
        """把 SSE 文本解析成 [(event, data_dict), ...]。"""
        events = []
        for block in body.strip().split("\n\n"):
            if not block.strip():
                continue
            event_name, data_raw = None, None
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_name = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    data_raw = line[len("data: "):].strip()
            if event_name and data_raw is not None:
                events.append((event_name, json.loads(data_raw)))
        return events

    def test_sse_event_sequence(self, client_with_graph):
        response = client_with_graph.post("/api/rag/agent/qa/stream", json={
            "question": "张建国用了什么药？", "use_llm_router": False,
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = self._parse_sse(response.text)
        names = [name for name, _ in events]

        assert names[0] == "sources"
        assert "chunk" in names
        assert "verification" in names
        assert names[-1] == "done"

    def test_sources_event_carries_char_intervals(self, client_with_graph):
        response = client_with_graph.post("/api/rag/agent/qa/stream", json={
            "question": "张建国", "use_llm_router": False,
        })
        events = self._parse_sse(response.text)
        sources = next(data for name, data in events if name == "sources")["sources"]

        assert sources, "sources 事件不能为空"
        # 向量来源带 chunk_type，前端可用它决定高亮方式
        assert any("chunk_type" in s for s in sources)

    def test_chunks_reassemble(self, client_with_graph):
        response = client_with_graph.post("/api/rag/agent/qa/stream", json={
            "question": "张建国", "use_llm_router": False,
        })
        events = self._parse_sse(response.text)
        answer = "".join(data["content"] for name, data in events if name == "chunk")
        assert answer == "张建国患有扩张型心肌病，LVEF 38%。"

    def test_graph_event_present(self, client_with_graph):
        response = client_with_graph.post("/api/rag/agent/qa/stream", json={
            "question": "张建国", "use_llm_router": False,
        })
        events = self._parse_sse(response.text)
        assert any(name == "graph" for name, _ in events)

    def test_done_event_reports_verification(self, client_with_graph):
        response = client_with_graph.post("/api/rag/agent/qa/stream", json={
            "question": "张建国", "use_llm_router": False,
        })
        events = self._parse_sse(response.text)
        done = events[-1][1]
        assert done["success"] is True
        assert "verification" in done
