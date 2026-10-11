"""Pure ASGI admission control: bounded work, no unbounded waiting queue.

Slots remain occupied until a streaming response finishes or disconnects.
All counters are per process; run exactly one API worker with the JSON graph store.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import Counter

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from app.config import Settings, get_settings
from app.security import get_auth_store

logger = logging.getLogger(__name__)

QUERY_POSTS = {
    "/api/rag/search", "/api/rag/qa", "/api/rag/qa/stream", "/api/rag/chat",
    "/api/rag/graph/query", "/api/rag/agent/qa", "/api/rag/agent/qa/stream",
}
PUBLIC_PATHS = {"/api/health", "/api/auth/login"}


class TrafficState:
    """Accessed exclusively on the ASGI event loop; no await inside counter mutations."""
    def __init__(self):
        self.active = 0
        self.expensive = 0
        self.auth_active = 0
        self.mutations = 0
        self.per_user = Counter()
        self.rejected = Counter()
        self.completed = Counter()
        self.windows = {}

    def rate_allowed(self, key: str, limit: int, now: float) -> bool:
        bucket = int(now // 60)
        # Bound memory under arbitrary source addresses; remove expired buckets first.
        if len(self.windows) >= 10000:
            self.windows = {k: v for k, v in self.windows.items() if v[0] == bucket}
            if key not in self.windows and len(self.windows) >= 10000:
                return False
        previous, count = self.windows.get(key, (bucket, 0))
        count = count if previous == bucket else 0
        if count >= limit:
            return False
        self.windows[key] = (bucket, count + 1)
        return True

    def snapshot(self):
        return {"active_requests": self.active, "active_expensive_requests": self.expensive,
                "active_mutations": self.mutations,
                "active_users": len(self.per_user), "completed": dict(self.completed),
                "rejected": dict(self.rejected), "scope": "single_process"}


class AccessMiddleware:
    def __init__(self, app):
        self.app = app
        self.state = TrafficState()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"].rstrip("/") or "/"
        method = scope["method"]
        settings = get_settings()
        scope.setdefault("state", {})["traffic"] = self.state
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        user_key = None
        expensive_slot = False
        auth_slot = False
        mutation_slot = False
        status = 500

        async def reject(code, detail, reason):
            nonlocal status
            status = code
            self.state.rejected[reason] += 1
            headers = {"X-Request-ID": request_id}
            if code in (429, 503):
                headers["Retry-After"] = "5" if code == 503 else str(60 - int(time.time()) % 60)
            if code == 401:
                headers["WWW-Authenticate"] = "Bearer"
            await JSONResponse({"detail": detail}, status_code=code, headers=headers)(scope, receive, send)

        # Keep health probes responsive under load. CORS middleware handles preflight outside us.
        if path == "/api/health" or not path.startswith("/api/"):
            return await self.app(scope, receive, send)
        if self.state.active >= settings.max_inflight_requests:
            return await reject(503, "Server busy; retry later", "inflight")
        self.state.active += 1
        try:
            if path == "/api/auth/login":
                # Never trust an arbitrary X-Forwarded-For from the client.
                peer = (scope.get("client") or ("unknown",))[0]
                if not self.state.rate_allowed(f"login:{peer}", settings.login_requests_per_minute, time.time()):
                    return await reject(429, "Too many login attempts", "login_rate")
                if self.state.auth_active >= settings.max_auth_concurrency:
                    return await reject(503, "Login capacity busy; retry later", "auth_capacity")
                self.state.auth_active += 1
                auth_slot = True
            if settings.auth_enabled and path not in PUBLIC_PATHS:
                auth = Headers(scope=scope).get("authorization", "")
                scheme, _, token = auth.partition(" ")
                user = None
                if scheme.lower() == "bearer" and token and len(token) <= 256:
                    user = await run_in_threadpool(lambda: get_auth_store(settings).authenticate(token))
                if not user:
                    return await reject(401, "Authentication required", "unauthorized")
                scope["state"]["user"] = user
                # All mutating routes require admin, unless explicitly listed as query/session routes.
                admin_only = path.startswith("/api/auth/users") or path == "/api/system/metrics"
                admin_only |= method not in ("GET", "HEAD", "OPTIONS") and path not in QUERY_POSTS | {"/api/auth/logout"}
                if admin_only and user["role"] != "admin":
                    return await reject(403, "Administrator role required", "forbidden")
                identity = user["id"]
            else:
                scope["state"]["user"] = {"id": "development", "username": "development", "role": "admin"}
                identity = "development"

            # Serialize shared KB management; extractors carry mutable last-run statistics.
            mutation = method not in ("GET", "HEAD", "OPTIONS") and path not in QUERY_POSTS | {"/api/auth/login", "/api/auth/logout"}
            if mutation:
                if self.state.mutations >= 1:
                    return await reject(503, "Knowledge-base management busy; retry later", "mutation_capacity")
                self.state.mutations += 1
                mutation_slot = True
            # Metadata/auth traffic does not consume costly LLM slots.
            expensive = path in QUERY_POSTS or path == "/api/extract" or path.startswith("/api/rag/pdf/") or (method == "POST" and path.startswith("/api/rag/") and path not in QUERY_POSTS)
            if expensive:
                if not self.state.rate_allowed(f"user:{identity}", settings.user_requests_per_minute, time.time()):
                    return await reject(429, "User request limit reached", "user_rate")
                if self.state.per_user[identity] >= settings.max_user_concurrency:
                    return await reject(429, "User concurrency limit reached", "user_concurrency")
                if self.state.expensive >= settings.max_expensive_requests:
                    return await reject(503, "Inference capacity busy; retry later", "expensive_capacity")
                user_key = identity
                expensive_slot = True
                self.state.expensive += 1
                self.state.per_user[identity] += 1

            async def tracked_send(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                    message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode()), (b"cache-control", b"no-store")]
                await send(message)

            await self.app(scope, receive, tracked_send)
        finally:
            self.state.active -= 1
            if auth_slot:
                self.state.auth_active -= 1
            if mutation_slot:
                self.state.mutations -= 1
            if expensive_slot:
                self.state.expensive -= 1
                self.state.per_user[user_key] -= 1
                if not self.state.per_user[user_key]:
                    del self.state.per_user[user_key]
            self.state.completed[str(status)] += 1
            logger.info("request_id=%s method=%s path=%s status=%d duration_ms=%.1f", request_id, method, path, status, (time.monotonic() - started) * 1000)


router = APIRouter(prefix="/system", tags=["Operations"])


@router.get("/metrics")
async def metrics(request: Request, settings: Settings = Depends(get_settings)):
    return {**request.state.traffic.snapshot(), "limits": {
        "inflight": settings.max_inflight_requests, "expensive": settings.max_expensive_requests,
        "per_user": settings.max_user_concurrency, "user_rpm": settings.user_requests_per_minute,
    }}
