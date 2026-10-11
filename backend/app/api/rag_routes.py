"""RAG 相关 API 路由 - PDF解析、语义搜索、智能问答"""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.utils.initialization import serialized_initialization
from app.models.schemas import (
    PDFParseRequest,
    PDFParseResponse,
    PDFTaskStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    QARequest,
    QAResponse,
    QASource,
    EntityInfo,
    ChatRequest,
    ChatResponse,
    AddDocumentRequest,
    AddDocumentResponse,
    AddExtractionsRequest,
    AddExtractionsResponse,
    VectorStoreStats,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# =============================================================================
# 全局服务实例
# =============================================================================
_pdf_parser = None
_vector_store = None
_qa_agent = None


@serialized_initialization
def get_pdf_parser(settings: Settings = Depends(get_settings)):
    """获取 PDF 解析器实例"""
    global _pdf_parser
    if _pdf_parser is None:
        from app.services.pdf_parser import PDFParser
        if settings.mineru_api_key:
            _pdf_parser = PDFParser(api_key=settings.mineru_api_key)
        else:
            raise HTTPException(
                status_code=500,
                detail="未配置 MINERU_API_KEY，无法使用 PDF 解析功能"
            )
    return _pdf_parser


@serialized_initialization
def get_vector_store(settings: Settings = Depends(get_settings)):
    """获取向量存储实例"""
    global _vector_store
    if _vector_store is None:
        if not settings.dashscope_api_key:
            raise HTTPException(
                status_code=500,
                detail="未配置 DASHSCOPE_API_KEY，无法使用向量存储功能"
            )

        # 根据配置选择向量存储后端
        backend = settings.vector_store_backend.lower()
        logger.info(f"使用向量存储后端: {backend}")

        if backend == "chroma":
            from app.services.vector_store_chroma import ChromaVectorStore
            _vector_store = ChromaVectorStore(
                collection_name="langextract_docs",
                persist_directory=settings.chroma_persist_dir,
                embedding_model=settings.embedding_model,
                embedding_api_key=settings.dashscope_api_key,
                embedding_base_url=settings.dashscope_base_url,
            )
            logger.info(f"ChromaDB 持久化目录: {settings.chroma_persist_dir}")
        else:
            # 默认使用 Qdrant
            from app.services.vector_store import VectorStore
            _vector_store = VectorStore(
                collection_name="langextract_docs",
                qdrant_url=settings.qdrant_url,
                qdrant_api_key=settings.qdrant_api_key,
                embedding_model=settings.embedding_model,
                embedding_api_key=settings.dashscope_api_key,
                embedding_base_url=settings.dashscope_base_url,
            )
            _vector_store.init_collection(recreate=False)

    return _vector_store


@serialized_initialization
def get_qa_agent(settings: Settings = Depends(get_settings)):
    """获取 QA Agent 实例"""
    global _qa_agent, _vector_store
    if _qa_agent is None:
        from app.services.qa_agent import QAAgent
        if settings.deepseek_api_key:
            # 确保向量存储已初始化
            vs = get_vector_store(settings)
            _qa_agent = QAAgent(
                vector_store=vs,
                model=settings.default_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.llm_timeout_seconds,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="未配置 DEEPSEEK_API_KEY，无法使用 QA 功能"
            )
    return _qa_agent


# =============================================================================
# PDF 解析 API
# =============================================================================

@router.post("/pdf/parse", response_model=PDFParseResponse, summary="解析 PDF 文档")
def parse_pdf(
    request: PDFParseRequest,
    settings: Settings = Depends(get_settings)
):
    """
    解析 PDF 文档并提取 Markdown 文本

    - **pdf_url**: PDF 文件的 URL 地址 (需要是可访问的 URL)
    - **model_version**: 模型版本，vlm 或 pipeline
    - **extract_after_parse**: 是否在解析后自动进行信息提取
    - **scenario**: 提取场景类型

    注意: MinerU API 不支持 github、aws 等国外 URL
    """
    try:
        parser = get_pdf_parser(settings)

        # 解析 PDF
        result = parser.parse(
            pdf_url=request.pdf_url,
            model_version=request.model_version,
            timeout=600
        )

        if not result["success"]:
            return PDFParseResponse(
                success=False,
                source=request.pdf_url,
                error=result.get("error", "解析失败")
            )

        extractions = []

        # 如果需要提取信息
        if request.extract_after_parse and request.scenario:
            from app.core.extractor import Extractor
            extractor = Extractor(settings)
            extract_result = extractor.extract(
                text=result["markdown"],
                scenario_id=request.scenario.value,
                use_cache=True
            )
            if extract_result["success"]:
                extractions = [
                    {
                        "extraction_class": ext["extraction_class"],
                        "extraction_text": ext["extraction_text"],
                        "attributes": ext.get("attributes"),
                        "char_interval": ext.get("char_interval"),
                    }
                    for ext in extract_result.get("extractions", [])
                ]

        return PDFParseResponse(
            success=True,
            task_id=result.get("task_id"),
            markdown=result.get("markdown"),
            source=request.pdf_url,
            parse_time=result.get("parse_time"),
            extractions=extractions,
        )

    except Exception as e:
        logger.error(f"PDF 解析失败: {e}")
        return PDFParseResponse(
            success=False,
            source=request.pdf_url,
            error=str(e)
        )


@router.post("/pdf/upload", response_model=PDFParseResponse, summary="上传并解析本地 PDF 文件")
def upload_and_parse_pdf(
    file: UploadFile = File(..., description="PDF 文件"),
    model_version: str = Form(default="vlm", description="模型版本 (vlm/pipeline)"),
    extract_after_parse: bool = Form(default=False, description="解析后是否进行信息提取"),
    scenario: Optional[str] = Form(default=None, description="提取场景类型"),
    settings: Settings = Depends(get_settings)
):
    """
    上传本地 PDF 文件并解析为 Markdown

    - **file**: PDF 文件 (最大 200MB，最多 600 页)
    - **model_version**: 模型版本，vlm 或 pipeline
    - **extract_after_parse**: 是否在解析后自动进行信息提取
    - **scenario**: 提取场景类型
    """
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return PDFParseResponse(
            success=False,
            source=file.filename or "unknown",
            error="请上传 PDF 文件"
        )

    # 检查文件大小 (最大 200MB)
    max_size = settings.max_upload_size
    content = file.file.read(settings.max_upload_size + 1)
    if len(content) > max_size:
        return PDFParseResponse(
            success=False,
            source=file.filename,
            error=f"文件过大，最大支持 200MB，当前 {len(content) / 1024 / 1024:.1f}MB"
        )

    try:
        parser = get_pdf_parser(settings)

        # 使用上传 API 解析
        result = parser.parse_uploaded_file(
            file_content=content,
            filename=file.filename,
            model_version=model_version,
            timeout=600
        )

        if not result["success"]:
            return PDFParseResponse(
                success=False,
                source=file.filename,
                error=result.get("error", "解析失败")
            )

        # 自动将解析后的文档添加到向量数据库，供智能问答使用
        try:
            vector_store = get_vector_store(settings)
            # 根据后端选择导入 DocumentChunk
            if settings.vector_store_backend.lower() == "chroma":
                from app.services.vector_store_chroma import DocumentChunk
            else:
                from app.services.vector_store import DocumentChunk
            import uuid as uuid_module

            doc_id = result.get("batch_id") or str(uuid_module.uuid4())
            markdown_text = result.get("markdown", "")

            # 按段落分块
            paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
            chunks = []
            for i, para in enumerate(paragraphs):
                chunk = DocumentChunk(
                    chunk_id=str(uuid_module.uuid4()),
                    doc_id=doc_id,
                    doc_title=file.filename,
                    content=para,
                    chunk_type="document",
                    attributes={"paragraph_index": i, "source": "pdf_upload"}
                )
                chunks.append(chunk)

            if chunks:
                vector_store.add_chunks(chunks)
                logger.info(f"已将 {len(chunks)} 个文档片段添加到向量数据库")
        except Exception as e:
            logger.warning(f"添加文档到向量数据库失败（不影响解析结果）: {e}")

        extractions = []

        # 如果需要提取信息
        if extract_after_parse and scenario:
            from app.core.extractor import Extractor
            extractor = Extractor(settings)
            extract_result = extractor.extract(
                text=result["markdown"],
                scenario_id=scenario,
                use_cache=True
            )
            if extract_result["success"]:
                extractions = [
                    {
                        "extraction_class": ext["extraction_class"],
                        "extraction_text": ext["extraction_text"],
                        "attributes": ext.get("attributes"),
                        "char_interval": ext.get("char_interval"),
                    }
                    for ext in extract_result.get("extractions", [])
                ]

        return PDFParseResponse(
            success=True,
            task_id=result.get("batch_id"),
            markdown=result.get("markdown"),
            source=file.filename,
            parse_time=result.get("parse_time"),
            extractions=extractions,
        )

    except Exception as e:
        logger.error(f"PDF 上传解析失败: {e}")
        return PDFParseResponse(
            success=False,
            source=file.filename or "unknown",
            error=str(e)
        )


@router.get("/pdf/task/{task_id}", response_model=PDFTaskStatusResponse, summary="查询 PDF 解析任务状态")
def get_pdf_task_status(
    task_id: str,
    settings: Settings = Depends(get_settings)
):
    """查询 PDF 解析任务的状态"""
    try:
        parser = get_pdf_parser(settings)
        result = parser.get_task_result(task_id)

        return PDFTaskStatusResponse(
            task_id=task_id,
            state=result.get("state", "unknown"),
            progress=result.get("extract_progress"),
            error=result.get("err_msg")
        )

    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        return PDFTaskStatusResponse(
            task_id=task_id,
            state="error",
            error=str(e)
        )


# =============================================================================
# 语义搜索 API
# =============================================================================

@router.post("/search", response_model=SearchResponse, summary="语义搜索")
def semantic_search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings)
):
    """
    语义搜索文档

    - **query**: 搜索查询文本
    - **top_k**: 返回结果数量 (1-20)
    - **doc_id**: 可选，限定在特定文档内搜索
    - **chunk_type**: 可选，限定片段类型
    """
    try:
        vector_store = get_vector_store(settings)

        results = vector_store.search(
            query=request.query,
            top_k=request.top_k,
            doc_id=request.doc_id,
            chunk_type=request.chunk_type
        )

        search_results = [
            SearchResult(
                score=r["score"],
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_title=r["doc_title"],
                content=r["content"],
                chunk_type=r["chunk_type"],
                attributes=r.get("attributes", {})
            )
            for r in results
        ]

        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total=len(search_results)
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return SearchResponse(
            success=False,
            query=request.query,
            error=str(e)
        )


# =============================================================================
# 智能问答 API
# =============================================================================

@router.post("/qa", response_model=QAResponse, summary="智能问答")
def question_answer(
    request: QARequest,
    settings: Settings = Depends(get_settings)
):
    """
    基于知识库的智能问答

    - **question**: 用户问题
    - **top_k**: 检索上下文数量 (1-10)
    - **system_prompt**: 可选的自定义系统提示词
    """
    try:
        qa_agent = get_qa_agent(settings)

        result = qa_agent.answer(
            question=request.question,
            top_k=request.top_k,
            system_prompt=request.system_prompt
        )

        if not result["success"]:
            return QAResponse(
                success=False,
                question=request.question,
                error=result.get("error", "问答失败")
            )

        sources = [
            QASource(
                doc_id=s["doc_id"],
                doc_title=s["doc_title"],
                content_preview=s["content_preview"],
                score=s["score"]
            )
            for s in result.get("sources", [])
        ]

        return QAResponse(
            success=True,
            question=request.question,
            answer=result.get("answer"),
            sources=sources,
            context_count=result.get("context_count", 0)
        )

    except Exception as e:
        logger.error(f"问答失败: {e}")
        return QAResponse(
            success=False,
            question=request.question,
            error=str(e)
        )


@router.post("/qa/stream", summary="智能问答（流式）")
def question_answer_stream(
    request: QARequest,
    settings: Settings = Depends(get_settings)
):
    """
    基于知识库的智能问答（流式输出）

    - **question**: 用户问题
    - **top_k**: 检索上下文数量 (1-10)
    - **system_prompt**: 可选的自定义系统提示词
    - **entities**: 已提取的实体列表（用于结构化展示）

    返回 SSE (Server-Sent Events) 流
    """
    import json

    # === 调试日志 ===
    logger.info("=" * 60)
    logger.info("[QA Stream] 收到请求")
    logger.info(f"[QA Stream] 问题: {request.question}")
    logger.info(f"[QA Stream] top_k: {request.top_k}")
    logger.info(f"[QA Stream] 前端传入实体数量: {len(request.entities) if request.entities else 0}")
    if request.entities:
        logger.info(f"[QA Stream] 前端传入实体列表: {request.entities[:5]}...")  # 只打印前5个
    logger.info("=" * 60)

    def event_generator():
        try:
            qa_agent = get_qa_agent(settings)

            # 首先发送检索结果（sources）
            logger.info("[QA Stream] 开始检索相关文档...")
            search_results = qa_agent.search_context(request.question, top_k=request.top_k)
            logger.info(f"[QA Stream] 检索到 {len(search_results)} 条结果")

            # 打印检索结果详情
            for i, s in enumerate(search_results):
                logger.info(f"[QA Stream] 检索结果[{i}]: doc_title={s.get('doc_title')}, score={s.get('score'):.4f}")
                logger.info(f"[QA Stream] 检索结果[{i}] 内容预览: {s.get('content', '')[:100]}...")

            sources = []
            for i, s in enumerate(search_results):
                attrs = s.get("attributes", {})
                # 解析 char_interval（可能是字符串或字典）
                char_interval = attrs.get("char_interval")
                if isinstance(char_interval, str):
                    try:
                        import json as json_module
                        char_interval = json_module.loads(char_interval)
                    except:
                        char_interval = None

                sources.append({
                    "doc_id": s.get("doc_id"),
                    "doc_title": s.get("doc_title"),
                    "content_preview": s.get("content", "")[:200],
                    "score": s.get("score", 0),
                    "chunk_index": attrs.get("paragraph_index", i),
                    "chunk_type": s.get("chunk_type", "text"),
                    "extraction_class": attrs.get("extraction_class"),
                    "char_interval": char_interval,
                    "attributes": attrs,
                })

            # 发送 sources 事件
            logger.info(f"[QA Stream] 发送 sources 事件，共 {len(sources)} 条")
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

            # 发送 entities 事件（前端传入的已提取实体）
            if request.entities:
                logger.info(f"[QA Stream] 发送 entities 事件，共 {len(request.entities)} 个实体")
                yield f"event: entities\ndata: {json.dumps(request.entities, ensure_ascii=False)}\n\n"

            # 流式生成答案
            logger.info("[QA Stream] 开始流式生成答案...")
            full_answer = ""
            chunk_count = 0
            for chunk in qa_agent.answer_stream(
                question=request.question,
                top_k=request.top_k,
                system_prompt=request.system_prompt
            ):
                full_answer += chunk
                chunk_count += 1
                # 发送 chunk 事件
                yield f"event: chunk\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

            logger.info(f"[QA Stream] 答案生成完成，共 {chunk_count} 个片段，总长度: {len(full_answer)} 字符")
            logger.info(f"[QA Stream] 完整答案（前1000字符）:\n{full_answer[:1000]}")
            logger.info(f"[QA Stream] 答案前10个字符: {repr(full_answer[:10])}")
            # 检查是否以 { 开头（JSON格式）
            is_json = full_answer.strip().startswith('{')
            logger.info(f"[QA Stream] 是否JSON格式: {is_json}")

            # 在答案生成完成后，匹配实体位置
            matched_entities = []
            if request.entities:
                logger.info(f"[QA Stream] 开始匹配实体位置...")
                for entity in request.entities:
                    entity_text = entity.get("text", "")
                    if entity_text and entity_text in full_answer:
                        # 找到所有出现的位置
                        start = 0
                        while True:
                            pos = full_answer.find(entity_text, start)
                            if pos == -1:
                                break
                            matched_entities.append({
                                "text": entity_text,
                                "entity_type": entity.get("entity_type", entity.get("extraction_class", "未知")),
                                "confidence": entity.get("confidence", 0.9),
                                "start_pos": pos,
                                "end_pos": pos + len(entity_text),
                            })
                            start = pos + 1
                logger.info(f"[QA Stream] 匹配到 {len(matched_entities)} 个实体")

            # 发送匹配到的实体位置
            if matched_entities:
                logger.info(f"[QA Stream] 发送 matched_entities 事件")
                yield f"event: matched_entities\ndata: {json.dumps(matched_entities, ensure_ascii=False)}\n\n"

            # 发送完成事件
            logger.info("[QA Stream] 发送 done 事件")
            yield f"event: done\ndata: {json.dumps({'success': True}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答失败: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/chat", response_model=ChatResponse, summary="多轮对话")
def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    支持多轮对话的智能问答

    - **messages**: 对话历史
    - **use_rag**: 是否使用 RAG 检索增强
    - **top_k**: 检索数量
    """
    try:
        qa_agent = get_qa_agent(settings)

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        result = qa_agent.chat(
            messages=messages,
            use_rag=request.use_rag,
            top_k=request.top_k
        )

        if not result["success"]:
            return ChatResponse(
                success=False,
                error=result.get("error", "对话失败")
            )

        sources = [
            QASource(
                doc_id=s["doc_id"],
                doc_title=s["doc_title"],
                content_preview=s["content_preview"],
                score=s["score"]
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(
            success=True,
            answer=result.get("answer"),
            sources=sources
        )

    except Exception as e:
        logger.error(f"对话失败: {e}")
        return ChatResponse(
            success=False,
            error=str(e)
        )


# =============================================================================
# 文档管理 API
# =============================================================================

@router.post("/documents", response_model=AddDocumentResponse, summary="添加文档到知识库")
def add_document(
    request: AddDocumentRequest,
    settings: Settings = Depends(get_settings)
):
    """
    添加文档到向量知识库

    - **doc_id**: 文档 ID，不传则自动生成
    - **title**: 文档标题
    - **content**: 文档内容
    - **metadata**: 可选的元数据
    """
    try:
        vector_store = get_vector_store(settings)

        doc_id = request.doc_id or str(uuid.uuid4())

        # 将文档分块并添加到向量存储
        # 根据后端选择导入 DocumentChunk
        if settings.vector_store_backend.lower() == "chroma":
            from app.services.vector_store_chroma import DocumentChunk
        else:
            from app.services.vector_store import DocumentChunk

        # 简单分块：按段落分割
        paragraphs = [p.strip() for p in request.content.split("\n\n") if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_title=request.title,
                content=para,
                chunk_type="document",
                attributes={
                    "paragraph_index": i,
                    **request.metadata
                }
            )
            chunks.append(chunk)

        # 如果没有段落，整个文档作为一个 chunk
        if not chunks:
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_title=request.title,
                content=request.content,
                chunk_type="document",
                attributes=request.metadata
            ))

        count = vector_store.add_chunks(chunks)

        return AddDocumentResponse(
            success=True,
            doc_id=doc_id,
            chunk_count=count
        )

    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return AddDocumentResponse(
            success=False,
            doc_id=request.doc_id or "",
            error=str(e)
        )


@router.post("/extractions", response_model=AddExtractionsResponse, summary="添加知识提取结果到知识库")
def add_extractions(
    request: AddExtractionsRequest,
    settings: Settings = Depends(get_settings)
):
    """
    添加知识提取结果到向量知识库（支持溯源）

    - **doc_id**: 文档 ID，不传则自动生成
    - **doc_title**: 文档标题
    - **extractions**: 知识提取结果列表，每项包含：
        - extraction_class: 提取类别（实体、关系描述、数据指标等）
        - extraction_text: 提取的文本内容
        - char_interval: 原文位置（start_pos, end_pos）
        - attributes: 属性信息（如关系类型、主体1、主体2等）
    """
    try:
        vector_store = get_vector_store(settings)
        doc_id = request.doc_id or str(uuid.uuid4())

        # 根据后端选择导入 DocumentChunk
        if settings.vector_store_backend.lower() == "chroma":
            from app.services.vector_store_chroma import DocumentChunk
        else:
            from app.services.vector_store import DocumentChunk

        chunks = []
        for i, ext in enumerate(request.extractions):
            # 构建属性（包含溯源信息）
            attributes = {
                "extraction_class": ext.extraction_class,
                "paragraph_index": i,
                **ext.attributes
            }

            # 添加 char_interval 到属性中
            if ext.char_interval:
                attributes["char_interval"] = {
                    "start_pos": ext.char_interval.start_pos,
                    "end_pos": ext.char_interval.end_pos
                }

            # 生成可搜索文本
            searchable_parts = [
                f"类型: {ext.extraction_class}",
                f"内容: {ext.extraction_text}",
                f"来源: {request.doc_title}"
            ]
            for k, v in ext.attributes.items():
                if k not in ["char_interval"]:
                    searchable_parts.append(f"{k}: {v}")

            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_title=request.doc_title,
                content=ext.extraction_text,
                chunk_type=ext.extraction_class,  # 使用 extraction_class 作为 chunk_type
                attributes=attributes
            )
            chunks.append(chunk)

        count = vector_store.add_chunks(chunks)
        logger.info(f"添加了 {count} 条知识提取结果到向量库，doc_id={doc_id}")

        return AddExtractionsResponse(
            success=True,
            doc_id=doc_id,
            chunk_count=count
        )

    except Exception as e:
        logger.error(f"添加知识提取结果失败: {e}")
        return AddExtractionsResponse(
            success=False,
            doc_id=request.doc_id or "",
            error=str(e)
        )


@router.delete("/documents/{doc_id}", summary="删除文档")
def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_settings)
):
    """删除指定文档及其所有向量"""
    try:
        vector_store = get_vector_store(settings)
        vector_store.delete_by_doc_id(doc_id)

        return {"success": True, "message": f"文档 {doc_id} 已删除"}

    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=VectorStoreStats, summary="获取知识库统计")
def get_vector_stats(
    settings: Settings = Depends(get_settings)
):
    """获取向量知识库的统计信息"""
    try:
        vector_store = get_vector_store(settings)
        info = vector_store.get_collection_info()

        return VectorStoreStats(
            collection_name=info.get("name", "unknown"),
            vectors_count=info.get("vectors_count", 0),
            points_count=info.get("points_count", 0)
        )

    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init", summary="初始化/重建知识库")
def init_vector_store(
    recreate: bool = False,
    settings: Settings = Depends(get_settings)
):
    """
    初始化向量知识库

    - **recreate**: 是否重建（删除所有已有数据）
    """
    try:
        vector_store = get_vector_store(settings)
        vector_store.init_collection(recreate=recreate)

        return {
            "success": True,
            "message": "知识库初始化完成" + ("（已重建）" if recreate else "")
        }

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
