from langchain_community.llms.tongyi import Tongyi
import os

# qwen-max 为大语言模型
model = Tongyi(model="qwen-max")

# 调用invoke向模型提问
res = model.invoke(input="请帮我写一首关于春天的诗歌")
print(res)
