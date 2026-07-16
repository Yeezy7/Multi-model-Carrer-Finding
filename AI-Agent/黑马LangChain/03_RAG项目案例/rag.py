from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class RagService(object):
    
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简介和专业的回答用户问题。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下:"), MessagesPlaceholder("history"),
                ("user", "请回答用户的问题：{input}")
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name, streaming=True)
        self.chain = self.__get_chain()
    
    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()  # 获取向量检索器

        def format_document(docs: list[Document]):
            if not docs:
                return "没有找到相关的参考资料"
            return "\n".join([doc.page_content for doc in docs])
        
        def print_prompt(prompt):
            print("="*10 + "提示词" + "="*10)
            print(prompt.to_string())
            print("="*20)
            return prompt

        def format_for_retriever(value: dict) -> str:
            print("--------", value)
            return value["input"]

        def format_for_prompt(value):
            print("--------", value)
            return {
                "input": value["input"]["input"], 
                "context": value["context"], 
                "history": value["input"]["history"]
            }
        
        chain = (
            {
                "input": RunnablePassthrough(), 
                "context": RunnableLambda(format_for_retriever) | retriever | format_document 
            } | RunnableLambda(format_for_prompt) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )
        
        conversation_chain = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=lambda session_id: FileChatMessageHistory(session_id, config.storage_path),
            input_messages_key="input",
            history_messages_key="history"
        )
        
        return conversation_chain
  
  
        
if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    rag_service = RagService()
    result = rag_service.chain.invoke({"input": "针织毛衣如何保养"}, config=session_config)
    print(result)   