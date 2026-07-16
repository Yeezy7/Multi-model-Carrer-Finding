from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf

class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """对话模型工厂类，用于生成对话模型实例"""
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"], streaming=True)
    

class EmbeddingsModelFactory(BaseModelFactory):
    """嵌入模型工厂类，用于生成嵌入模型实例"""
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 实例化模型对象，供业务代码直接使用
chat_model = ChatModelFactory().generate()
embedding_model = EmbeddingsModelFactory().generate()