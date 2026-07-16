from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi


prompt_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

# 示例的动态数据注入，要求是list内部套字典
examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "高", "antonym": "低"}
]

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=prompt_template,   # 示例数据的模板
    examples=examples_data,         # 示例数据列表
    prefix="告知我单词的反义词，如下示例：",           # 提示词前缀
    suffix="基于前面的示例告知我，{input_word}的反义词是？",           # 提示词后缀
    input_variables=["input_word"],    # 输入变量列表
)

prompt_text = few_shot_prompt.invoke(input={"input_word": "快"}).to_string()  # 调用.format方法注入信息即可

# print(prompt_text)

model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)
