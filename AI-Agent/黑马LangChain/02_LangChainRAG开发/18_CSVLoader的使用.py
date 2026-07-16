from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="AI-Agent/黑马LangChain/data/stu.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": "|",  # 分隔符
        "quotechar": '"',  # 引号字符
        "fieldnames": ["姓名", "年龄", "性别", "爱好"]  # 字段名  如果原数据有表头，不要设置这个参数
    }
)

# 批量加载 .load() -> [Document, Document, ...]
# documents = loader.load()
# for document in documents:
#     print(document)
#     print(type(document))


# 懒加载 .lazy_loader 迭代器[Document, Document, ...]
for document in loader.lazy_load():
    print(document)