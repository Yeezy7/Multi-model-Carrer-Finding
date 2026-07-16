from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(
    file_path="AI-Agent/黑马LangChain/data/pdf2.pdf",
    # mode="page",         # 默认是page模式，每个页面形成一个Document文档对象,
    mode="single",         # single模式：整个pdf形成一个Document文档对象
    password="itheima"     # 密码，默认为空字符串
    
)

for doc in loader.lazy_load():
    print(doc)