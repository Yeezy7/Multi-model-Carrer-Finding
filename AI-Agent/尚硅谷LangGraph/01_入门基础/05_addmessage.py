from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import add_messages

left = [
    SystemMessage(content="你是一个专业的翻译", id='1'),
    HumanMessage(content="你好，我是你爸爸", id='2'),
    AIMessage(content="你好，我是AI翻译", id='3'),
    AIMessage(content="我是人工智障", id='4'),
    AIMessage(content="我是傻逼", id='5')
]

right = [
    HumanMessage(content="你好，我是你爹", id='2'),
    AIMessage(content="好的，我记住了", id='3'),
    HumanMessage(content="你是谁？", id='4'),

]

result = add_messages(left, right)
print(result)