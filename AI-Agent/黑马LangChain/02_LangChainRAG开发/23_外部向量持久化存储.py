from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# Chroma 向量数据库（轻量级）
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="my_collection",    # 当前向量存储起个名字，类似数据库的表名称
    embedding_function=DashScopeEmbeddings(),    # 指定向量存储使用的向量化模型
    persist_directory="./chroma_db",  # 指定向量存储的持久化存储路径
)

loader = CSVLoader(
    file_path="AI-Agent/黑马LangChain/data/info.csv",
    encoding="utf-8",
    source_column="source",      # 指定本条数据的来源列
)

documents = loader.load()
# print(documents[0])

# 向量存储的 新增、删除、检索
# 新增
vector_store.add_documents(
    documents=documents,                # 被添加的文档，类型是: list[Document]
    ids=[f"id{i}" for i in range(1, len(documents)+1)]  # 指定每条数据的唯一标识符
)

# 删除
vector_store.delete(ids=["id1", "id2"])  # 指定要删除的文档的唯一标识符

# 检索
result = vector_store.similarity_search(
    query="Python是不是简单易学呀",
    k=3 ,                    # 检索的数量
    filter={"source": "黑马程序员"}    # 挀索的文档来源必须是黑马程序员
)
print(result)
