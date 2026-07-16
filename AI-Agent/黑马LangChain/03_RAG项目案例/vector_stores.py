from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    """向量存储服务类：负责从本地存储中检索向量数据，返回向量检索器"""
    def __init__(self, embedding):
        """
        Args:
            embedding (_type_): 嵌入模型的传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
    
    def get_retriever(self, ):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})  # 返回向量检索器，k表示返回的相似文档数量

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    embedding = DashScopeEmbeddings(model="text-embedding-v4")
    vector_store_service = VectorStoreService(embedding)
    retriever = vector_store_service.get_retriever()
    result = retriever.invoke("我的体重180斤，尺码推荐")
    print(result)