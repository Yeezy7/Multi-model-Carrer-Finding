from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    "AI-Agent/黑马LangChain/data/python基础语法.txt",
    encoding="utf-8",
    
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 每个文档块的最大长度
    chunk_overlap=20,  # 文档块之间的重叠长度
    separators=["\n\n", "\n", " ", "", "。", "!", "?", "！", "？", "，", "；", "："],  # 分割符列表，按优先级顺序
    length_function=len  # 用于计算文本长度的函数，默认使用len函数
)

split_docs = text_splitter.split_documents(docs)  # 将文档分割成多个块
print(len(split_docs))  # 输出分割后的文档块数量
for doc in split_docs:
    print("="*20)
    # print(doc)
    print(doc.page_content)
    print("="*20)
