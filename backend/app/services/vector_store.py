"""向量存储服务 - 基于 Qdrant"""

import os
import uuid
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """文档片段"""
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    chunk_type: str = "text"  # text, entity, relationship
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_searchable_text(self) -> str:
        """生成用于向量化的可搜索文本"""
        parts = [
            f"类型: {self.chunk_type}",
            f"内容: {self.content}",
            f"来源: {self.doc_title}"
        ]
        if self.attributes:
            for k, v in self.attributes.items():
                parts.append(f"{k}: {v}")
        return " | ".join(parts)


class VectorStore:
    """向量存储管理器 - 基于 Qdrant"""

    def __init__(
        self,
        collection_name: str = "documents",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = None,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
    ):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            qdrant_url: Qdrant 服务器 URL，留空使用内存模式
            qdrant_api_key: Qdrant API Key
            embedding_model: 嵌入模型名称
            embedding_api_key: 嵌入模型 API Key (DashScope)
            embedding_base_url: 嵌入模型 API Base URL
        """
        self.collection_name = collection_name

        # 从环境变量获取配置
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.embedding_api_key = embedding_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.embedding_base_url = embedding_base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 初始化 Embeddings
        if not self.embedding_api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，无法初始化 Embeddings")

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            check_embedding_ctx_length=False,
            chunk_size=10,  # 阿里云百炼限制批量大小不超过 10
        )
        logger.info(f"使用嵌入模型: {self.embedding_model}")

        # 初始化 Qdrant 客户端
        if self.qdrant_url:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                timeout=30.0
            )
            logger.info(f"连接到 Qdrant 服务器: {self.qdrant_url}")
        else:
            self.client = QdrantClient(":memory:")
            logger.info("使用 Qdrant 内存模式")

        # 获取向量维度
        self._vector_size = self._get_vector_size()
        logger.info(f"向量维度: {self._vector_size}")

    def _get_vector_size(self) -> int:
        """获取 embedding 向量维度"""
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)

    def init_collection(self, recreate: bool = False):
        """
        初始化集合

        Args:
            recreate: 是否重建集合（删除旧数据）
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            if recreate:
                self.client.delete_collection(self.collection_name)
                logger.info(f"已删除旧集合: {self.collection_name}")
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                return

        # 创建新集合
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"已创建集合: {self.collection_name}")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_id_field: str = "doc_id",
        content_field: str = "content",
        title_field: str = "title"
    ) -> int:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表
            doc_id_field: 文档 ID 字段名
            content_field: 内容字段名
            title_field: 标题字段名

        Returns:
            添加的记录数
        """
        if not documents:
            return 0

        chunks = []
        for doc in documents:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc.get(doc_id_field, str(uuid.uuid4())),
                doc_title=doc.get(title_field, "未知文档"),
                content=doc.get(content_field, ""),
                chunk_type="document",
                attributes={k: v for k, v in doc.items() if k not in [doc_id_field, content_field, title_field]}
            )
            chunks.append(chunk)

        return self.add_chunks(chunks)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        添加文档片段到向量存储

        Args:
            chunks: 文档片段列表

        Returns:
            添加的记录数
        """
        if not chunks:
            return 0

        # 准备数据
        texts = [chunk.to_searchable_text() for chunk in chunks]

        # 生成向量（分批处理）
        logger.info(f"生成 {len(texts)} 个向量...")
        vectors = self.embeddings.embed_documents(texts)

        # 准备 points
        points = []
        for chunk, vector in zip(chunks, vectors):
            point = models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "attributes": chunk.attributes,
                    "searchable_text": chunk.to_searchable_text()
                }
            )
            points.append(point)

        # 批量插入
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"已添加 {len(points)} 个向量")
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            chunk_type: 过滤的片段类型
            doc_id: 过滤的文档 ID
            score_threshold: 相关度阈值

        Returns:
            搜索结果列表
        """
        # 生成查询向量
        query_vector = self.embeddings.embed_query(query)

        # 构建过滤器
        must_conditions = []
        if chunk_type:
            must_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type)
                )
            )
        if doc_id:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            )

        query_filter = None
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)

        # 搜索
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=score_threshold
        )

        # 格式化结果
        formatted_results = []
        for hit in results.points:
            formatted_results.append({
                "score": hit.score,
                "chunk_id": hit.id,
                "doc_id": hit.payload.get("doc_id"),
                "doc_title": hit.payload.get("doc_title"),
                "content": hit.payload.get("content"),
                "chunk_type": hit.payload.get("chunk_type"),
                "attributes": hit.payload.get("attributes", {}),
            })

        return formatted_results

    def get_collection_info(self) -> dict:
        """获取集合信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "error": str(e)
            }

    def get_kb_summary(self) -> dict:
        """获取知识库概况：文档数量、片段数量、文档列表"""
        try:
            info = self.client.get_collection(self.collection_name)
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            doc_map = {}
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id") or "unknown"
                title = payload.get("doc_title") or "未知文档"
                doc_map.setdefault(doc_id, title)
            documents = [{"doc_id": doc_id, "title": title} for doc_id, title in doc_map.items()]
            return {
                "collection_name": self.collection_name,
                "chunk_count": info.points_count,
                "document_count": len(documents),
                "documents": documents,
            }
        except Exception as e:
            logger.error(f"获取知识库概况失败: {e}")
            return {
                "collection_name": self.collection_name,
                "chunk_count": 0,
                "document_count": 0,
                "documents": [],
                "error": str(e),
            }

    def delete_collection(self):
        """删除集合"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        根据文档 ID 删除所有相关向量

        Args:
            doc_id: 文档 ID

        Returns:
            删除的数量
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"已删除文档 {doc_id} 的所有向量")
            return 1  # Qdrant 不返回删除数量
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return 0
