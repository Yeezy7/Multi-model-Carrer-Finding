from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max")

res = model.stream(input="你是谁啊，你能做什么")

for chunk in res:
    print(chunk, end="", flush=True)
