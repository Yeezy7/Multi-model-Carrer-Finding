from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

# 创建所需的解析器
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# 创建模型
model = ChatTongyi(model="qwen3-max")

# 第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我领居姓{lastname}, 刚生了{gender}, 请帮我起个名字，并封装为json格式。"
    "要求key是name，value是你起的名字，请严格遵守格式要求。"

)

# 第二个提示词模板
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我分析这个名字的含义和寓意"
)

# 创建链条
# chain = first_prompt | model | json_parser
# res = chain.invoke({"lastname": "范", "gender": "女儿"})

# print(res)
# print(type(res))

# invoke
# chain = first_prompt | model | json_parser | second_prompt | model | str_parser
# res = chain.invoke({"lastname": "范", "gender": "女儿"})

# print(res)
# print(type(res))

# stream
chain = first_prompt | model | json_parser | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "范", "gender": "儿子"}):
    print(chunk, end="", flush=True)