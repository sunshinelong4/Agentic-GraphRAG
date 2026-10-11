"""FastAPI 应用入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import get_settings
from app.api.routes import router
from app.api.rag_routes import router as rag_router
from app.api.graph_routes import router as graph_router, agent_router
from app.security import router as auth_router
from app.traffic import AccessMiddleware, router as operations_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info(f"Starting Agentic-GraphRAG with model: {settings.default_model}")
    logger.info(f"API endpoint: {settings.deepseek_base_url}")
    logger.info(f"知识图谱持久化路径: {settings.graph_persist_path}")

    # 检查 API Key
    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY not set!")
    if not settings.dashscope_api_key:
        logger.warning("DASHSCOPE_API_KEY not set! 向量检索不可用，Agent 将只能使用图谱检索")

    yield

    logger.info("Shutting down Agentic-GraphRAG")


# 创建 FastAPI 应用
app = FastAPI(
    title="Agentic-GraphRAG API",
    description=(
        "面向垂直领域文档的图谱增强智能问答平台 —— "
        "LangExtract 结构化抽取 + Chroma 向量检索 + 后端知识图谱多跳检索 + LangGraph Agent"
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置 CORS
settings = get_settings()
app.add_middleware(AccessMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api")
app.include_router(rag_router, prefix="/api")
# Agentic-GraphRAG 新增：知识图谱接口与 Agentic 问答接口
app.include_router(graph_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(operations_router, prefix="/api")


def session_openapi():
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerSession"] = {
            "type": "http", "scheme": "bearer", "description": "Login via /api/auth/login, then paste access_token here."
        }
        for path, operations in schema["paths"].items():
            if path.startswith("/api/") and path not in {"/api/health", "/api/auth/login"}:
                for method, operation in operations.items():
                    if method in {"get", "post", "put", "patch", "delete", "head", "options"}:
                        operation["security"] = [{"BearerSession": []}]
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = session_openapi


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Agentic-GraphRAG",
        "version": "0.2.0",
        "description": "图谱增强的垂直领域智能问答平台",
        "docs": "/docs",
        "capabilities": {
            "extraction": "/api/extract",
            "vector_rag": "/api/rag/qa",
            "knowledge_graph": "/api/rag/graph/query",
            "agent_qa": "/api/rag/agent/qa",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
