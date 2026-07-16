from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我领居姓{lastname}, 刚生了{gender}, 请帮我起个名字，无需其他内容"
)
parser = StrOutputParser()

chain = prompt | model | parser | model | parser

res = chain.invoke({"lastname": "范", "gender": "男"})

# print(res.content)
print(res)
