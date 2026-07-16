from langchain_community.document_loaders import JSONLoader

# 单个json
# loader = JSONLoader(
#     file_path="AI-Agent/黑马LangChain/data/stu.json",
#     jq_schema=".",
#     text_content=False,  # 告知JSONLoader 抽取的内容不是字符串
# )

# documents = loader.load()
# print(documents)

# # json数组
# loader = JSONLoader(
#     file_path="AI-Agent/黑马LangChain/data/stus.json",
#     jq_schema=".[].name",  # [0] 表示抽取第一个元素的name字段 ; 不填下标，表示抽取所有元素的name字段，
#     text_content=False,  # 告知JSONLoader 抽取的内容不是字符串
# )

# documents = loader.load()
# print(documents)



# 多个json  JSONLines文件
loader = JSONLoader(
    file_path="AI-Agent/黑马LangChain/data/stu_json_lines.json",
    jq_schema=".name",  # [0] 表示抽取第一个元素的name字段 ; 不填下标，表示抽取所有元素的name字段，
    text_content=False,  # 告知JSONLoader 抽取的内容不是字符串
    json_lines=True  # 告知JSONLoader 这是一个JSONLines文件
)

documents = loader.load()
print(documents)