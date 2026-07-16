from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),
    ]
)

history_data = [
    ("human", "请给我一首唐诗"),
    ("ai", "好的，这是一首唐诗：床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    ("human", "请再来一首唐诗"),
    ("ai", "好的，这是一首唐诗：春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。")
]

# StringPromptValue
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
# print(prompt_text)

model = ChatTongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res.content, type(res))
