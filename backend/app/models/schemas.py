"""Pydantic 数据模型定义"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    """场景类型"""
    RADIOLOGY = "radiology"      # 放射学报告
    MEDICATION = "medication"    # 药物信息
    NEWS = "news"                # 新闻信息
    FINANCE = "finance"          # 金融分析
    MEDICAL = "medical"          # 医疗病历
    SALES = "sales"              # 销售商机
    CUSTOMER_SERVICE = "customer_service"  # 客服工单
    CUSTOM = "custom"            # 自定义


class CharInterval(BaseModel):
    """字符位置区间"""
    start_pos: int = Field(..., description="起始位置")
    end_pos: int = Field(..., description="结束位置")


class ExtractionItem(BaseModel):
    """单个提取项"""
    extraction_class: str = Field(..., description="提取类别")
    extraction_text: str = Field(..., description="提取文本")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="属性")
    char_interval: Optional[CharInterval] = Field(default=None, description="字符位置")


class SegmentInfo(BaseModel):
    """段落信息 (用于放射学报告等分段场景)"""
    type: str = Field(..., description="段落类型")
    label: Optional[str] = Field(default=None, description="段落标签")
    content: str = Field(..., description="段落内容")
    intervals: List[CharInterval] = Field(default_factory=list, description="字符位置列表")
    significance: Optional[str] = Field(default=None, description="重要性")


class ExtractionRequest(BaseModel):
    """提取请求"""
    text: str = Field(..., min_length=1, max_length=100000, description="输入文本")  # 增加到 100000 以支持长文档
    scenario: ScenarioType = Field(default=ScenarioType.RADIOLOGY, description="场景类型")
    use_cache: bool = Field(default=True, description="是否使用缓存")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "患者李明，男，58岁，CT检查发现肺部结节...",
                "scenario": "radiology",
                "use_cache": True
            }
        }


class ExtractionResponse(BaseModel):
    """提取响应"""
    success: bool = Field(..., description="是否成功")
    scenario: ScenarioType = Field(..., description="场景类型")
    segments: List[SegmentInfo] = Field(default_factory=list, description="分段结果")
    extractions: List[ExtractionItem] = Field(default_factory=list, description="提取结果")
    formatted_text: str = Field(default="", description="格式化后的文本")
    sanitized_input: str = Field(default="", description="预处理后的输入")
    from_cache: bool = Field(default=False, description="是否来自缓存")
    error: Optional[str] = Field(default=None, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="状态")
    version: str = Field(..., description="版本")
    model: str = Field(..., description="使用的模型")


class ScenarioInfo(BaseModel):
    """场景信息"""
    id: str = Field(..., description="场景ID")
    name: str = Field(..., description="场景名称")
    description: str = Field(..., description="场景描述")
    extract_classes: List[str] = Field(..., description="提取类别列表")


class SampleReport(BaseModel):
    """样本报告"""
    id: str = Field(..., description="样本ID")
    title: str = Field(..., description="标题")
    scenario: ScenarioType = Field(..., description="场景类型")
    text: str = Field(..., description="文本内容")


# =============================================================================
# PDF 解析相关模型
# =============================================================================

class PDFParseRequest(BaseModel):
    """PDF 解析请求"""
    pdf_url: str = Field(..., description="PDF 文件 URL")
    model_version: str = Field(default="vlm", description="模型版本 (vlm/pipeline)")
    extract_after_parse: bool = Field(default=True, description="解析后是否进行信息提取")
    scenario: Optional[ScenarioType] = Field(default=None, description="提取场景类型")

    class Config:
        json_schema_extra = {
            "example": {
                "pdf_url": "https://example.com/document.pdf",
                "model_version": "vlm",
                "extract_after_parse": True,
                "scenario": "news"
            }
        }


class PDFParseResponse(BaseModel):
    """PDF 解析响应"""
    success: bool = Field(..., description="是否成功")
    task_id: Optional[str] = Field(default=None, description="任务 ID")
    markdown: Optional[str] = Field(default=None, description="解析后的 Markdown 文本")
    source: str = Field(..., description="PDF 来源")
    parse_time: Optional[float] = Field(default=None, description="解析耗时(秒)")
    extractions: List[ExtractionItem] = Field(default_factory=list, description="提取结果")
    error: Optional[str] = Field(default=None, description="错误信息")


class PDFTaskStatusResponse(BaseModel):
    """PDF 任务状态响应"""
    task_id: str = Field(..., description="任务 ID")
    state: str = Field(..., description="任务状态")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="进度信息")
    error: Optional[str] = Field(default=None, description="错误信息")


# =============================================================================
# 语义搜索相关模型
# =============================================================================

class SearchRequest(BaseModel):
    """语义搜索请求"""
    query: str = Field(..., min_length=1, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    doc_id: Optional[str] = Field(default=None, description="限定文档 ID")
    chunk_type: Optional[str] = Field(default=None, description="限定片段类型")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "民间借贷利率",
                "top_k": 5
            }
        }


class SearchResult(BaseModel):
    """搜索结果项"""
    score: float = Field(..., description="相关度分数")
    chunk_id: str = Field(..., description="片段 ID")
    doc_id: str = Field(..., description="文档 ID")
    doc_title: str = Field(..., description="文档标题")
    content: str = Field(..., description="内容")
    chunk_type: str = Field(..., description="片段类型")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")


class SearchResponse(BaseModel):
    """语义搜索响应"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果")
    total: int = Field(default=0, description="结果总数")
    error: Optional[str] = Field(default=None, description="错误信息")


# =============================================================================
# 智能问答相关模型
# =============================================================================

class QARequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=10, description="检索数量")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="已提取的实体列表（用于结构化展示）")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "什么是民间借贷的利率上限？",
                "top_k": 5,
                "entities": [
                    {"text": "民间借贷", "entity_type": "法律概念"},
                    {"text": "利率上限", "entity_type": "法律术语"}
                ]
            }
        }


class QASource(BaseModel):
    """问答来源引用"""
    doc_id: str = Field(..., description="文档 ID")
    doc_title: str = Field(..., description="文档标题")
    content_preview: str = Field(..., description="内容预览")
    score: float = Field(..., description="相关度分数")


class EntityInfo(BaseModel):
    """实体信息（用于结构化展示）"""
    text: str = Field(..., description="实体文本")
    entity_type: str = Field(..., description="实体类型")
    confidence: float = Field(default=0.9, description="置信度")
    start_pos: Optional[int] = Field(default=None, description="在答案中的起始位置")
    end_pos: Optional[int] = Field(default=None, description="在答案中的结束位置")


class QAResponse(BaseModel):
    """问答响应"""
    success: bool = Field(..., description="是否成功")
    question: str = Field(..., description="用户问题")
    answer: Optional[str] = Field(default=None, description="回答内容")
    sources: List[QASource] = Field(default_factory=list, description="引用来源")
    context_count: int = Field(default=0, description="检索到的上下文数量")
    entities: List[EntityInfo] = Field(default_factory=list, description="答案中的实体信息")
    error: Optional[str] = Field(default=None, description="错误信息")


class ChatMessage(BaseModel):
    """对话消息"""
    role: str = Field(..., description="角色 (user/assistant)")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""
    messages: List[ChatMessage] = Field(..., min_length=1, description="对话历史")
    use_rag: bool = Field(default=True, description="是否使用 RAG 检索")
    top_k: int = Field(default=5, ge=1, le=10, description="检索数量")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "请介绍一下劳动合同法的相关规定"}
                ],
                "use_rag": True,
                "top_k": 5
            }
        }


class ChatResponse(BaseModel):
    """对话响应"""
    success: bool = Field(..., description="是否成功")
    answer: Optional[str] = Field(default=None, description="回答内容")
    sources: List[QASource] = Field(default_factory=list, description="引用来源")
    error: Optional[str] = Field(default=None, description="错误信息")


# =============================================================================
# 文档管理相关模型
# =============================================================================

class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str = Field(..., description="文档 ID")
    title: str = Field(..., description="标题")
    content_preview: str = Field(..., description="内容预览")
    chunk_count: int = Field(default=0, description="片段数量")
    created_at: Optional[str] = Field(default=None, description="创建时间")


class AddDocumentRequest(BaseModel):
    """添加文档请求"""
    doc_id: Optional[str] = Field(default=None, description="文档 ID，不传则自动生成")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., min_length=1, description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "示例文档",
                "content": "这是文档的内容...",
                "metadata": {"source": "manual", "type": "legal"}
            }
        }


class KnowledgeExtractionItem(BaseModel):
    """知识提取项（用于存储到向量库）"""
    extraction_class: str = Field(..., description="提取类别（实体、关系描述、数据指标等）")
    extraction_text: str = Field(..., description="提取的文本内容")
    char_interval: Optional[CharInterval] = Field(default=None, description="原文位置")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性信息")


class AddExtractionsRequest(BaseModel):
    """添加知识提取结果请求"""
    doc_id: Optional[str] = Field(default=None, description="文档 ID，不传则自动生成")
    doc_title: str = Field(..., description="文档标题")
    extractions: List[KnowledgeExtractionItem] = Field(..., min_length=1, description="知识提取结果列表")

    class Config:
        json_schema_extra = {
            "example": {
                "doc_title": "中医药研究报告",
                "extractions": [
                    {
                        "extraction_class": "实体",
                        "extraction_text": "丹参",
                        "char_interval": {"start_pos": 100, "end_pos": 102},
                        "attributes": {"类型": "中药", "功效": "活血化瘀"}
                    },
                    {
                        "extraction_class": "关系描述",
                        "extraction_text": "丹参能够抑制肝星状细胞活化",
                        "char_interval": {"start_pos": 200, "end_pos": 215},
                        "attributes": {"类型": "作用机制", "主体1": "丹参", "关系": "抑制", "主体2": "肝星状细胞"}
                    }
                ]
            }
        }


class AddExtractionsResponse(BaseModel):
    """添加知识提取结果响应"""
    success: bool = Field(..., description="是否成功")
    doc_id: str = Field(..., description="文档 ID")
    chunk_count: int = Field(default=0, description="创建的向量数量")
    error: Optional[str] = Field(default=None, description="错误信息")


class AddDocumentResponse(BaseModel):
    """添加文档响应"""
    success: bool = Field(..., description="是否成功")
    doc_id: str = Field(..., description="文档 ID")
    chunk_count: int = Field(default=0, description="创建的片段数量")
    error: Optional[str] = Field(default=None, description="错误信息")


class VectorStoreStats(BaseModel):
    """向量存储统计"""
    collection_name: str = Field(..., description="集合名称")
    vectors_count: int = Field(default=0, description="向量数量")
    points_count: int = Field(default=0, description="点数量")


# =============================================================================
# 知识图谱相关模型（Agentic-GraphRAG 新增）
# =============================================================================

class GraphBuildRequest(BaseModel):
    """从原文 + 抽取结果构建知识图谱"""
    doc_id: Optional[str] = Field(default=None, description="文档 ID，不传则自动生成")
    doc_title: str = Field(..., description="文档标题")
    text: str = Field(..., min_length=1, description="原文（用于定位关系证据的字符区间）")
    extractions: List[KnowledgeExtractionItem] = Field(
        ..., min_length=1, description="实体抽取结果（来自 /api/extract 或 /api/rag/pdf/upload）"
    )
    build_relations: bool = Field(default=True, description="是否调用 LLM 抽取关系（关闭则只建实体节点）")
    max_relations: int = Field(default=60, ge=1, le=200, description="关系数上限")

    class Config:
        json_schema_extra = {
            "example": {
                "doc_title": "病历.pdf",
                "text": "患者张建国，诊断为扩张型心肌病。给予沙库巴曲缬沙坦钠片治疗。",
                "extractions": [
                    {"extraction_class": "患者", "extraction_text": "张建国",
                     "char_interval": {"start_pos": 2, "end_pos": 5}},
                    {"extraction_class": "诊断", "extraction_text": "扩张型心肌病",
                     "char_interval": {"start_pos": 10, "end_pos": 16}}
                ],
                "build_relations": True
            }
        }


class GraphBuildResponse(BaseModel):
    """图谱构建响应"""
    success: bool = Field(..., description="是否成功")
    doc_id: str = Field(default="", description="文档 ID")
    entities_added: int = Field(default=0, description="新增实体数")
    entities_merged: int = Field(default=0, description="归并到已有实体的数量")
    relations_added: int = Field(default=0, description="新增关系数")
    relations_merged: int = Field(default=0, description="合并的重复关系数")
    rejected: int = Field(default=0, description="被校验拒绝的关系数")
    graph_stats: Dict[str, Any] = Field(default_factory=dict, description="图谱整体统计")
    relation_extraction: Dict[str, Any] = Field(default_factory=dict, description="关系抽取过程统计")
    error: Optional[str] = Field(default=None, description="错误信息")


class GraphQueryRequest(BaseModel):
    """图谱检索请求"""
    query: str = Field(..., min_length=1, description="查询文本")
    mode: str = Field(default="auto", description="检索模式：auto / local / global")
    hops: int = Field(default=2, ge=1, le=4, description="多跳扩展跳数")
    max_nodes: int = Field(default=30, ge=1, le=100, description="子图节点上限")

    class Config:
        json_schema_extra = {
            "example": {"query": "张建国用的药是哪家公司研发的", "mode": "auto", "hops": 3}
        }


class GraphQueryResponse(BaseModel):
    """图谱检索响应"""
    success: bool = Field(..., description="是否成功")
    mode: str = Field(default="local", description="实际使用的检索模式")
    hit: bool = Field(default=False, description="是否命中图谱内容")
    seed_count: int = Field(default=0, description="实体链接命中的种子数")
    node_count: int = Field(default=0, description="子图节点数")
    edge_count: int = Field(default=0, description="子图边数")
    chain_count: int = Field(default=0, description="关系链数量")
    seeds: List[Dict[str, Any]] = Field(default_factory=list, description="命中的种子实体")
    context_text: str = Field(default="", description="可直接喂给 LLM 的上下文文本")
    error: Optional[str] = Field(default=None, description="错误信息")


class GraphStatsResponse(BaseModel):
    """图谱统计响应"""
    success: bool = Field(..., description="是否成功")
    entity_count: int = Field(default=0, description="实体数")
    relation_count: int = Field(default=0, description="关系数")
    document_count: int = Field(default=0, description="文档数")
    mention_count: int = Field(default=0, description="提及总数")
    traceable_ratio: float = Field(default=0.0, description="可溯源提及占比")
    avg_degree: float = Field(default=0.0, description="平均度数")
    community_count: int = Field(default=0, description="社区数量")
    modularity: float = Field(default=0.0, description="社区划分的模块度 Q")
    entity_type_distribution: Dict[str, int] = Field(default_factory=dict, description="实体类型分布")
    persist_path: str = Field(default="", description="图谱持久化路径")
    error: Optional[str] = Field(default=None, description="错误信息")


class CommunityInfo(BaseModel):
    """社区信息"""
    id: str = Field(..., description="社区 ID（内部字典序最小的成员）")
    size: int = Field(..., description="社区规模")
    entity_type_distribution: Dict[str, int] = Field(default_factory=dict, description="实体类型分布")
    core_entities: List[Dict[str, Any]] = Field(default_factory=list, description="核心实体（按度数）")
    relation_types: List[str] = Field(default_factory=list, description="社区内关系类型")
    members: List[Dict[str, Any]] = Field(default_factory=list, description="全部成员实体")


class CommunitiesResponse(BaseModel):
    """社区列表响应"""
    success: bool = Field(..., description="是否成功")
    community_count: int = Field(default=0, description="社区数量")
    modularity: float = Field(default=0.0, description="模块度 Q（>0.3 通常认为社区结构显著）")
    communities: List[CommunityInfo] = Field(default_factory=list, description="社区列表")
    error: Optional[str] = Field(default=None, description="错误信息")


class EntityDetailResponse(BaseModel):
    """实体详情响应"""
    success: bool = Field(..., description="是否成功")
    found: bool = Field(default=False, description="是否找到该实体")
    entity: Dict[str, Any] = Field(default_factory=dict, description="实体信息（含提及与字符区间）")
    neighbors: List[Dict[str, Any]] = Field(default_factory=list, description="N 跳邻居")
    relations: List[Dict[str, Any]] = Field(default_factory=list, description="关联关系（含证据与区间）")
    error: Optional[str] = Field(default=None, description="错误信息")


class AgentQARequest(BaseModel):
    """Agent 问答请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    hops: int = Field(default=2, ge=1, le=4, description="图检索跳数")
    top_k: int = Field(default=5, ge=1, le=10, description="向量检索召回数")
    max_attempts: int = Field(default=2, ge=1, le=3, description="反思重试的最大轮数")
    max_tool_rounds: int = Field(default=3, ge=1, le=5, description="工具调用轮数上限（多轮 ReAct 循环）")
    use_llm_router: bool = Field(default=True, description="是否用 LLM 复核检索规划")
    use_llm_assessor: bool = Field(default=True, description="是否用 LLM 判断证据充分性（不足则补充检索）")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "张建国用的药是哪家公司研发的？",
                "hops": 3,
                "max_tool_rounds": 3,
                "max_attempts": 2,
            }
        }


class AgentQAResponse(BaseModel):
    """Agent 问答响应"""
    success: bool = Field(..., description="是否成功")
    question: str = Field(..., description="用户问题")
    answer: str = Field(default="", description="回答内容")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源（含字符区间）")
    route: List[str] = Field(default_factory=list, description="本次累计调用的检索工具")
    graph: Dict[str, Any] = Field(default_factory=dict, description="图检索结果")
    verification: Dict[str, Any] = Field(default_factory=dict, description="引用校验结果")
    attempts: int = Field(default=1, description="反思重试的实际轮数")
    tool_rounds: int = Field(default=1, description="实际执行的工具调用轮数")
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM 用量：调用次数、token 数、按节点归因的耗时；配置单价后含费用估算",
    )
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="节点执行轨迹")
    error: Optional[str] = Field(default=None, description="错误信息")
