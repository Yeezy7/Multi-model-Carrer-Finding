"""
提示词：用户的提问 + 向量库中检索到的参考资料
"""

from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# 检索
results = vector_store.search(
    query=input_text, 
    search_type="similarity", 
    k=3
)

# print(result)

def print_prompt(prompt):
    print("="*10 + "提示词" + "="*10)
    print(prompt.to_string())
    print("="*20)
    return prompt

chain = prompt | print_prompt | model | StrOutputParser()
response = chain.invoke({
    "context": {result.page_content for result in results},
    "input": input_text
})
print(response)