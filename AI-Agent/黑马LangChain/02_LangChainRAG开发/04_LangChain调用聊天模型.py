from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 得到模型对象
model = ChatTongyi(model="qwen3-max")

# 准备消息列表
messages = [
    SystemMessage(content="你是一个专业的助手"),
    HumanMessage(content="你是谁，你的数据库截止到什么时候？"),
    AIMessage(content="我是你爹"),
    HumanMessage(content="我操你妈"),
]

# 调用stream流式执行
res = model.stream(messages)

# for循环迭代打印
for chunk in res:
    print(chunk.content, end="", flush=True)