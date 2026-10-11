"""QA Agent 服务 - 基于 LangChain"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# 知识库元信息问题识别（文档数量/列表/统计等）
KB_META_PATTERNS = [
    r"几个文档", r"多少文档", r"几份文档", r"多少份文档", r"几篇文档", r"多少篇文档",
    r"有哪些文档", r"包含哪些文档", r"文档列表", r"文档清单", r"文档名",
    r"知识库.*(统计|概况|数量|多少|几个|有哪些|包含哪些|是不是空|是否为空|是空的|为空|有没有内容|有什么内容)",
    r"(统计|概况|数量).*知识库",
]

KB_META_SYSTEM_PROMPT = (
    "你是知识库管理助手。请根据给定的【知识库概况】如实、准确地回答文档数量、文档名称和内容概况。\n"
    "要求：\n"
    "1. 只依据【知识库概况】提供的信息回答，不要编造不存在的文档；\n"
    "2. 如果知识库为空，请明确告知“知识库当前为空，还没有任何文档”；\n"
    "3. 回答时列出文档名称，并可简要概括每个文档的内容；\n"
    "4. 不要提及“来源”“参考”等检索性词语。"
)


class QAAgent:
    """智能问答代理 - 基于 LangChain"""

    def __init__(
        self,
        vector_store=None,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
        timeout: float = 45,
    ):
        """
        初始化 QA Agent

        Args:
            vector_store: 向量存储实例
            model: 模型名称
            api_key: API Key
            base_url: API Base URL
            temperature: 温度参数
        """
        self.vector_store = vector_store

        # 从环境变量获取配置
        self.model = model or os.getenv("DEFAULT_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not self.api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY")

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=1,
        )

        logger.info(f"QA Agent 初始化完成，使用模型: {self.model}")

    def set_vector_store(self, vector_store):
        """设置向量存储"""
        self.vector_store = vector_store

    def search_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关上下文

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        if not self.vector_store:
            return []

        results = self.vector_store.search(query, top_k=top_k)
        return results

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果为上下文文本

        Args:
            search_results: 搜索结果

        Returns:
            格式化的上下文文本
        """
        if not search_results:
            return "未找到相关信息。"

        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            doc_title = result.get("doc_title", "未知来源")
            score = result.get("score", 0)

            context_parts.append(
                f"[{i}] 来源: {doc_title} (相关度: {score:.2f})\n"
                f"内容: {content}\n"
            )

        return "\n".join(context_parts)

    def build_prompt(
        self,
        question: str,
        context: str,
        system_prompt: str = None,
        structured: bool = True
    ) -> List[dict]:
        """
        构建对话提示

        Args:
            question: 用户问题
            context: 上下文信息
            system_prompt: 系统提示词
            structured: 是否输出结构化格式

        Returns:
            消息列表
        """
        if not system_prompt:
            # 使用结构化文本格式（非JSON），便于前端解析
            system_prompt = """你是一个专业的知识问答助手。请用简洁的结构化格式回答问题。

输出格式要求：
1. 先用一句话总结答案
2. 然后分点列出主要内容，每点用"• "开头
3. 每个要点包含：标题、关键词（用【】标注）、详细说明
4. 最后可以加一个简短结论

重要：不要在回答中提及"来源"、"文档"、"参考"等词，直接陈述知识内容。"""

        user_content = f"""请根据以下信息回答问题。

参考信息：
{context}

问题：{question}

请按格式要求回答（不要提及来源）："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        logger.info(f"[build_prompt] structured={structured}")
        logger.info(f"[build_prompt] system_prompt: {system_prompt[:100]}...")
        logger.info(f"[build_prompt] user_content 长度: {len(user_content)}")

        return messages

    def _is_kb_meta_question(self, question: str) -> bool:
        """判断问题是否为知识库元信息问题（文档数量/列表/统计等）"""
        if not question:
            return False
        return any(re.search(pattern, question) for pattern in KB_META_PATTERNS)

    def _build_kb_summary_text(self) -> str:
        """从向量库读取真实的知识库概况文本"""
        if not self.vector_store:
            return "知识库当前为空，还没有任何文档。"
        try:
            summary = self.vector_store.get_kb_summary()
        except Exception as e:
            logger.warning(f"获取知识库概况失败: {e}")
            return "知识库概况暂时无法获取，请稍后再试。"
        doc_count = summary.get("document_count", 0)
        chunk_count = summary.get("chunk_count", 0)
        if doc_count == 0:
            return "知识库当前为空，还没有任何文档。"
        documents = summary.get("documents", []) or []
        title_list = "、".join(f"《{d.get('title', '未知文档')}》" for d in documents) or "未知"
        return (
            "【知识库概况】\n"
            f"共 {doc_count} 个文档，共 {chunk_count} 个片段。\n"
            f"文档列表：{title_list}。"
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: str = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        回答问题

        Args:
            question: 用户问题
            top_k: 检索数量
            system_prompt: 自定义系统提示词
            return_sources: 是否返回引用来源

        Returns:
            回答结果
        """
        try:
            # 0. 知识库元信息问题：直接读取向量库统计，不走相似度检索
            if self._is_kb_meta_question(question):
                search_results = []
                context = self._build_kb_summary_text()
                messages = self.build_prompt(question, context, KB_META_SYSTEM_PROMPT)
            else:
                # 1. 检索相关文档
                search_results = self.search_context(question, top_k=top_k)

                # 2. 格式化上下文
                context = self.format_context(search_results)

                # 3. 构建提示
                messages = self.build_prompt(question, context, system_prompt)

            # 4. 调用 LLM
            response = self.llm.invoke([
                SystemMessage(content=messages[0]["content"]),
                HumanMessage(content=messages[1]["content"])
            ])

            answer_text = response.content

            # 5. 提取引用来源
            sources = []
            if return_sources and search_results:
                for result in search_results:
                    sources.append({
                        "doc_id": result.get("doc_id"),
                        "doc_title": result.get("doc_title"),
                        "content_preview": result.get("content", "")[:200],
                        "score": result.get("score", 0),
                    })

            return {
                "success": True,
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "context_count": len(search_results),
            }

        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {
                "success": False,
                "question": question,
                "answer": None,
                "error": str(e),
            }

    def answer_stream(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: str = None
    ):
        """
        流式回答问题（生成器）

        Args:
            question: 用户问题
            top_k: 检索数量
            system_prompt: 自定义系统提示词

        Yields:
            回答内容的增量片段
        """
        try:
            # 0. 知识库元信息问题：直接读取向量库统计，不走相似度检索
            if self._is_kb_meta_question(question):
                search_results = []
                context = self._build_kb_summary_text()
                messages = self.build_prompt(question, context, KB_META_SYSTEM_PROMPT)
            else:
                # 1. 检索相关文档
                search_results = self.search_context(question, top_k=top_k)

                # 2. 格式化上下文
                context = self.format_context(search_results)

                # 3. 构建提示
                messages = self.build_prompt(question, context, system_prompt)

            # 4. 流式调用 LLM
            from langchain_core.messages import HumanMessage, SystemMessage

            for chunk in self.llm.stream([
                SystemMessage(content=messages[0]["content"]),
                HumanMessage(content=messages[1]["content"])
            ]):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"流式问答失败: {e}")
            yield f"错误: {str(e)}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        多轮对话

        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            use_rag: 是否使用 RAG 检索
            top_k: 检索数量

        Returns:
            回答结果
        """
        try:
            # 获取最后一条用户消息作为查询
            last_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break

            if not last_user_message:
                return {
                    "success": False,
                    "error": "没有找到用户消息"
                }

            # 检索相关上下文
            context = ""
            search_results = []
            if use_rag and self.vector_store and self._is_kb_meta_question(last_user_message):
                # 知识库元信息问题：直接使用向量库统计
                context = self._build_kb_summary_text()
                system_content = KB_META_SYSTEM_PROMPT + "\n\n" + context
            else:
                if use_rag and self.vector_store:
                    search_results = self.search_context(last_user_message, top_k=top_k)
                    context = self.format_context(search_results)

                # 构建消息列表
                system_content = """你是一个专业的知识问答助手。基于提供的上下文信息回答用户问题。
如果上下文信息不足，请明确告知用户。回答时引用信息来源。"""

                if context:
                    system_content += f"\n\n【参考上下文】\n{context}"

            llm_messages = [SystemMessage(content=system_content)]

            for msg in messages:
                if msg["role"] == "user":
                    llm_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    llm_messages.append(AIMessage(content=msg["content"]))

            # 调用 LLM
            response = self.llm.invoke(llm_messages)

            sources = []
            for result in search_results:
                sources.append({
                    "doc_id": result.get("doc_id"),
                    "doc_title": result.get("doc_title"),
                    "content_preview": result.get("content", "")[:200],
                    "score": result.get("score", 0),
                })

            return {
                "success": True,
                "answer": response.content,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"对话失败: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# 单例实例
_qa_agent_instance: Optional[QAAgent] = None


def get_qa_agent(vector_store=None) -> QAAgent:
    """获取 QA Agent 单例"""
    global _qa_agent_instance

    if _qa_agent_instance is None:
        _qa_agent_instance = QAAgent(vector_store=vector_store)
    elif vector_store is not None:
        _qa_agent_instance.set_vector_store(vector_store)

    return _qa_agent_instance
