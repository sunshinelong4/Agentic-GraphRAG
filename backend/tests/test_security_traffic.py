"""Exercise authentication, RBAC, actual overlapping requests and SSE slot cleanup offline."""
import asyncio
import sqlite3
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.security import AuthStore, get_auth_store, password_matches, router
from app.traffic import AccessMiddleware, TrafficState


@pytest.fixture
def protected_app(workspace_tmp, monkeypatch):
    settings = Settings(auth_db_path=f"{workspace_tmp}/auth.sqlite3", auth_enabled=True,
                        user_requests_per_minute=1000, login_requests_per_minute=1000)
    monkeypatch.setattr("app.traffic.get_settings", lambda: settings)
    store = get_auth_store(settings)
    store.create_user("admin", "admin-password-123", "admin")
    store.create_user("reader", "reader-password-123", "user")
    app = FastAPI()
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(router, prefix="/api")
    app.add_middleware(AccessMiddleware)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/rag/documents")
    async def document():
        return {"success": True}

    return app, settings, store


def authenticate(client, username="reader", password="reader-password-123"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_auth_rbac_revocation_and_no_password_leak(protected_app):
    app, _, store = protected_app
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={"username": "reader", "password": "wrong"}).status_code == 401
        reader = authenticate(client)
        assert client.get("/api/auth/me", headers=reader).json()["role"] == "user"
        assert client.post("/api/rag/documents", headers=reader).status_code == 403
        assert client.post("/api/auth/users", headers=reader, json={"username": "evil", "password": "password-1234", "role": "admin"}).status_code == 403
        assert client.get("/api/auth/users", headers=reader).status_code == 403
        admin = authenticate(client, "admin", "admin-password-123")
        assert client.post("/api/rag/documents", headers=admin).status_code == 200
        created = client.post("/api/auth/users", headers=admin, json={"username": "newuser", "password": "new-password-123"})
        assert created.status_code == 201
        assert "password" not in str(created.json())
        assert client.post("/api/auth/users", headers=admin, json={"username": "newuser", "password": "new-password-123"}).status_code == 409
        assert client.post("/api/auth/logout", headers=reader).status_code == 204
        assert client.get("/api/auth/me", headers=reader).status_code == 401
    with store.connect() as db:
        raw = dict(db.execute("SELECT * FROM users WHERE username='reader'").fetchone())
        assert raw["password_hash"] != "reader-password-123"
        assert password_matches("reader-password-123", raw["password_hash"])


def test_expired_and_disabled_sessions_fail_closed(protected_app):
    app, _, store = protected_app
    with TestClient(app) as client:
        headers = authenticate(client)
        with store.connect() as db:
            db.execute("UPDATE sessions SET expires_at=0")
        assert client.get("/api/auth/me", headers=headers).status_code == 401
        headers = authenticate(client)
        with store.connect() as db:
            db.execute("UPDATE users SET active=0 WHERE username='reader'")
        assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_login_rate_limit(protected_app):
    app, settings, _ = protected_app
    settings.login_requests_per_minute = 1
    with TestClient(app) as client:
        authenticate(client)
        response = client.post("/api/auth/login", json={"username": "reader", "password": "wrong"})
        assert response.status_code == 429
        assert "Retry-After" in response.headers


@pytest.mark.parametrize("user_limit,capacity,rejected_status", [(10, 2, 503), (2, 10, 429)])
def test_parallel_requests_are_bounded_and_health_is_available(monkeypatch, user_limit, capacity, rejected_status):
    settings = Settings(auth_enabled=False, max_expensive_requests=capacity, max_user_concurrency=user_limit,
                        user_requests_per_minute=10000)
    monkeypatch.setattr("app.traffic.get_settings", lambda: settings)

    async def scenario():
        app = FastAPI()
        entered = 0
        peak = 0
        both_entered = asyncio.Event()
        release = asyncio.Event()

        @app.post("/api/rag/agent/qa")
        async def qa():
            nonlocal entered, peak
            entered += 1
            peak = max(peak, entered)
            if entered == 2:
                both_entered.set()
            await release.wait()
            entered -= 1
            return {"success": True}

        @app.get("/api/health")
        async def health():
            return {"status": "ok"}

        middleware = AccessMiddleware(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
            tasks = [asyncio.create_task(client.post("/api/rag/agent/qa")) for _ in range(20)]
            await asyncio.wait_for(both_entered.wait(), 2)
            health = await asyncio.wait_for(client.get("/api/health"), 1)
            assert health.status_code == 200
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            assert peak == 2
            assert sum(r.status_code == 200 for r in results) == 2
            assert sum(r.status_code == rejected_status for r in results) == 18
            assert middleware.state.active == middleware.state.expensive == 0
            assert not middleware.state.per_user
            assert (await client.post("/api/rag/agent/qa")).status_code == 200

    asyncio.run(scenario())


def test_sse_slot_is_held_until_disconnect_and_released(monkeypatch):
    settings = Settings(auth_enabled=False, max_expensive_requests=1, user_requests_per_minute=1000)
    monkeypatch.setattr("app.traffic.get_settings", lambda: settings)

    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"data: chunk\n\n", "more_body": True})
            started.set()
            await finish.wait()
            raise asyncio.CancelledError()  # client disconnect

        middleware = AccessMiddleware(downstream)
        scope = {"type": "http", "method": "POST", "path": "/api/rag/agent/qa/stream", "headers": [], "client": ("test", 1)}
        async def receive():
            return {"type": "http.disconnect"}
        async def send(message):
            pass
        task = asyncio.create_task(middleware(scope, receive, send))
        await asyncio.wait_for(started.wait(), 2)
        assert middleware.state.expensive == 1
        rejected = []
        async def second_send(message):
            rejected.append(message)
        await middleware(dict(scope, state={}), receive, second_send)
        assert rejected[0]["status"] == 503
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert middleware.state.expensive == middleware.state.active == 0

    asyncio.run(scenario())


def test_fixed_window_rollover_and_key_capacity():
    state = TrafficState()
    assert state.rate_allowed("u", 1, 0)
    assert not state.rate_allowed("u", 1, 59)
    assert state.rate_allowed("u", 1, 60)
    state.windows = {str(i): (1, 1) for i in range(10000)}
    assert not state.rate_allowed("new", 1, 60)
    assert state.rate_allowed("new", 1, 120)


def test_agent_requests_do_not_share_mutable_parameters(monkeypatch):
    from app.api import graph_routes
    from app.models.schemas import AgentQARequest
    from app.services.graph_store import KnowledgeGraphStore

    store = KnowledgeGraphStore()
    store.upsert_entity("Example", "organization")
    monkeypatch.setattr(graph_routes, "get_graph_store", lambda settings: store)
    monkeypatch.setattr(graph_routes, "_get_vector_store", lambda settings: None)
    first = graph_routes._build_agent(Settings())
    second = graph_routes._build_agent(Settings())
    graph_routes._apply_agent_params(first, AgentQARequest(question="question", hops=1, top_k=1))
    assert second.top_k == 5
    assert first.graph_retriever is not second.graph_retriever
    store.clear()
    assert first.graph_store.entities
    assert second.graph_store.entities


def test_cache_parallel_writes_and_return_value_isolation(workspace_tmp):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from app.utils.cache import get_cache_manager

    cache = get_cache_manager(workspace_tmp, True)
    assert cache is get_cache_manager(workspace_tmp, True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: cache.set(str(index), "test", {"index": index, "items": []}), range(40)))
    raw = json.loads((Path(workspace_tmp) / "extraction_cache.json").read_text(encoding="utf-8"))
    assert len(raw) == 40
    result = cache.get("0", "test")
    result["items"].append("changed")
    assert cache.get("0", "test")["items"] == []


def test_server_agent_budget_cannot_be_increased(monkeypatch):
    from app.api import graph_routes
    from app.models.schemas import AgentQARequest
    from app.services.graph_store import KnowledgeGraphStore

    monkeypatch.setattr(graph_routes, "get_graph_store", lambda settings: KnowledgeGraphStore())
    monkeypatch.setattr(graph_routes, "_get_vector_store", lambda settings: None)
    agent = graph_routes._build_agent(Settings(agent_max_attempts=1, agent_max_tool_rounds=1))
    graph_routes._apply_agent_params(agent, AgentQARequest(question="question", max_attempts=3, max_tool_rounds=5))
    assert agent.max_attempts == agent.max_tool_rounds == 1


def test_openapi_has_bearer_security_without_locking_health():
    from app.main import app
    schema = app.openapi()
    assert schema["components"]["securitySchemes"]["BearerSession"]["scheme"] == "bearer"
    assert schema["paths"]["/api/rag/agent/qa"]["post"]["security"] == [{"BearerSession": []}]
    assert "security" not in schema["paths"]["/api/health"]["get"]
