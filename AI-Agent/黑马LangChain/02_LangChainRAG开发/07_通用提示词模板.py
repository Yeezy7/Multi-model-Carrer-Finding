from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的领居姓{lastname}, 刚生了{gender}, 你能帮起个名字吗？ "
)

# 调用.format方法注入信息即可
prompt_text = prompt_template.format(lastname="张", gender="男孩")
# print(prompt_template.format(lastname="张", gender="男孩"))

model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)
