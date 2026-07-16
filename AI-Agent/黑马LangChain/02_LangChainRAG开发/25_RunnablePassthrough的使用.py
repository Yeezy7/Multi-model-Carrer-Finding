"""
提示词：用户的提问 + 向量库中检索到的参考资料
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

def print_prompt(prompt):
    print("="*10 + "提示词" + "="*10)
    print(prompt.to_string())
    print("="*20)
    return prompt

def print_context(context):
    print("="*10 + "检索结果" + "="*10)
    print(context)
    print("="*20)
    return context


model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "以我提供的已知参考资料为主，简介和专业的回答用户问题。参考资料：{context}。"),
        ("user", "用户提问：{input}")   
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 准备数据
vector_store.add_texts(["减肥就是要少吃多练", "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦", "减肥需要长期坚持"])

input_text = "怎么减肥？"

# langchain中向量存储对象，有一个方法：as_retriever()，可以将向量存储对象转换为检索器对象
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def format_func(docs: list[Document]) -> str:
    if not docs:
        return "无相关参考资料"
    return "".join([doc.page_content for doc in docs])
    # formatted_str = ""
    # for doc in docs:
    #     formatted_str += doc.page_content
    # return formatted_str


# chain = retriever | prompt | print_prompt | model | StrOutputParser()
"""
retriever:
    输入：用户的提问        str
    输出：向量库的检索结果   list[Document]
prompt:
    输入：用户的提问 + 向量库的检索结果  dict
    输出：完整的提示词                 PromptValue
"""
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func | print_context} | prompt | print_prompt | model | StrOutputParser())

response = chain.invoke(input_text)
print(response)