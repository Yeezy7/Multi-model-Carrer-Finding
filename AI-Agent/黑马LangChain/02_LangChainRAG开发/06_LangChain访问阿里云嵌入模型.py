from langchain_community.embeddings import DashScopeEmbeddings

# 创建模型，不传模型名称默认使用 text-embedding-v1
model = DashScopeEmbeddings()

# 不用invok stream
# print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "我不喜欢你"]))