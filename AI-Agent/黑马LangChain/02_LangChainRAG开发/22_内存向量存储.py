from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

# 向量存储对象  
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(),
)

loader = CSVLoader(
    file_path="AI-Agent/黑马LangChain/data/info.csv",
    encoding="utf-8",
    source_column="source",     # 指定本条数据的来源列
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
result = vector_store.search(
    query="Python是不是简单易学呀",
    search_type="similarity",
    k=3                     # 检索的数量
)
print(result)
