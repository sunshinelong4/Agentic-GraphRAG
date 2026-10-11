"""应用配置管理"""

from functools import lru_cache
from typing import List
import os

# backend 目录的绝对路径（用于构造不受启动目录影响的存储路径）
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FILE = os.path.join(BACKEND_DIR, ".env")

# 用**绝对路径**加载 .env。
#
# 修复的 bug：原代码是 `load_dotenv("../.env")`，这是相对**当前工作目录**解析的。
# 从 backend/ 启动时它指向项目根目录（那里并没有 .env），
# 于是 os.getenv 全部读到空值；而 pydantic 的 env_file=".env" 恰好按 CWD
# 找到了 backend/.env，导致"一部分配置能读到、一部分读不到"的诡异现象。
# 更要命的是从项目根启动（uvicorn backend.app.main:app）会**全部读不到**。
from dotenv import load_dotenv

load_dotenv(ENV_FILE, override=False)

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置"""

    # DeepSeek API 配置
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com"
    default_model: str = os.getenv("DEFAULT_MODEL")

    # MinerU API 配置 (PDF 解析)
    mineru_api_key: str = ""

    # 阿里云百炼 Embedding 配置
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = os.getenv("EMBEDDING_MODEL")

    # Qdrant 向量数据库配置
    qdrant_url: str = ""  # 留空使用内存模式
    qdrant_api_key: str = ""

    # ChromaDB 向量数据库配置
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BACKEND_DIR, "chroma_db_chapter5"))
    vector_store_backend: str = "chroma"  # chroma 或 qdrant

    # 知识图谱配置（Agentic-GraphRAG）
    # 用 BACKEND_DIR 拼绝对路径，避免像 cache_dir 那样依赖启动时的工作目录
    graph_persist_path: str = os.getenv(
        "GRAPH_PERSIST_PATH", os.path.join(BACKEND_DIR, "outputs", "knowledge_graph_store.json")
    )
    graph_hops: int = int(os.getenv("GRAPH_HOPS", "2"))            # 多跳检索默认跳数
    graph_max_nodes: int = int(os.getenv("GRAPH_MAX_NODES", "30"))  # 子图节点预算
    graph_max_relations: int = int(os.getenv("GRAPH_MAX_RELATIONS", "60"))  # 单文档关系数上限
    agent_max_attempts: int = int(os.getenv("AGENT_MAX_ATTEMPTS", "2"))     # 反思重试轮数上限
    agent_grounded_threshold: float = float(os.getenv("AGENT_GROUNDED_THRESHOLD", "0.85"))
    # 工具调用轮数上限（多轮 ReAct 循环）：每多一轮都可能多一次规划/判官 LLM 调用
    agent_max_tool_rounds: int = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "3"))
    # 是否用 LLM 判断证据充分性。关掉则退化为纯启发式（确定、零额外调用）
    agent_use_llm_assessor: bool = os.getenv("AGENT_USE_LLM_ASSESSOR", "true").lower() in ("1", "true", "yes")
    # 证据"薄"的判定阈值（字符数）：低于该值才请 LLM 判官复核
    agent_thin_evidence_chars: int = int(os.getenv("AGENT_THIN_EVIDENCE_CHARS", "400"))

    # LLM 单价（用于成本估算，单位：元 / 百万 token）
    # 刻意不写死默认值：各家模型单价会变，代码里写死迟早变成错误信息。
    # 未配置时只报 token 数，cost 返回 null 并标记 cost_configured=false。
    llm_price_input_per_1m: float = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0"))
    llm_price_output_per_1m: float = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "0"))

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS 配置
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # 缓存配置
    cache_enabled: bool = True
    cache_dir: str = "./cache"

    # 文件上传配置
    upload_dir: str = "./uploads"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # 输入限制
    max_input_length: int = 10000

    # Shared knowledge base: authentication is on by default. Bootstrap via python -m app.security.
    auth_enabled: bool = True
    auth_db_path: str = os.path.join(BACKEND_DIR, "outputs", "auth.sqlite3")
    auth_session_ttl_seconds: int = Field(default=28800, ge=60, le=604800)
    max_inflight_requests: int = Field(default=64, ge=1, le=1000)
    max_expensive_requests: int = Field(default=8, ge=1, le=32)
    max_user_concurrency: int = Field(default=2, ge=1, le=32)
    user_requests_per_minute: int = Field(default=30, ge=1)
    login_requests_per_minute: int = Field(default=10, ge=1)
    max_auth_concurrency: int = Field(default=2, ge=1, le=8)
    llm_timeout_seconds: float = Field(default=45, ge=1, le=300)

    @property
    def cors_origins_list(self) -> List[str]:
        """返回 CORS 源列表"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ENV_FILE  # 绝对路径，不受启动目录影响
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
