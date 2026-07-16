from langchain_community.chat_models.tongyi import ChatTongyi

# 得到模型对象
model = ChatTongyi(model="qwen3-max")

# 准备消息列表
messages = [
    ("system", "你是一个专业的助手"),
    ("user", "你是谁，你的数据库截止到什么时候？"),
    ("assistant", "我是你爹"),
    ("user", "我操你妈"),
]

# 调用stream流式执行
res = model.stream(messages)

# for循环迭代打印
for chunk in res:
    print(chunk.content, end="", flush=True)