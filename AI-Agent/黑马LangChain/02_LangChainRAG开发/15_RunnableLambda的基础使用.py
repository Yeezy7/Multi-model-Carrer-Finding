from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

# 创建所需的解析器
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# 创建模型
model = ChatTongyi(model="qwen3-max")

# 第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我领居姓{lastname}, 刚生了{gender}, 请帮我起个名字。仅告知我名字，不需要其他信息。"
)

# 第二个提示词模板
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我分析这个名字的含义和寓意"
)

# 函数的入参：AIMessages -> 字典dict {"name": "xxx"}
name_parser = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

# chain = first_prompt | model | name_parser | second_prompt | model | str_parser

chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "范", "gender": "儿子"}):
    print(chunk, end="", flush=True)