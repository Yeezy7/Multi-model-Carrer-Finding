"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_summarize_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.logger_handler import logger


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_summarize_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()  # 初始化链
    
    def print_prompt(self, prompt):
        logger.info(f"[print_prompt] 当前提示词:\n{prompt}")
        return prompt
    
    def _init_chain(self):
        chain = self.prompt_template | self.print_prompt | self.model | StrOutputParser()
        return chain
    
    def retriever_documents(self, question: str) -> list[Document]:
        """检索相关文档"""
        return self.retriever.invoke(question)
    
    def rag_summarize(self, question: str) -> str:
        """RAG总结"""
        context_docs = self.retriever_documents(question)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】:\n{doc.page_content}\n | 参考元数据: {doc.metadata}\n\n"
        logger.info(f"[rag_summarize] 检索到 {counter} 条相关文档。\n 内容如下：{context}")
        return self.chain.invoke({
                "input": question,
                "context": context,
            }
        )

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    question = "小户型适合哪些扫地机器人"
    answer = rag_service.rag_summarize(question)
    print(answer)
        