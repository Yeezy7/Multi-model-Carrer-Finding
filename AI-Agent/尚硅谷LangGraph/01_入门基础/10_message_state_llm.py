import langchain
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import MessagesState
from typing import TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi


model = ChatTongyi(model="qwen3-max")

# 1. 定义状态
class OverAllState(MessagesState):
    """全局状态"""
    username:str
    output: str

# 2, 定义节点
def node_a(state: OverAllState) -> OverAllState:
    return {
        "messages": [HumanMessage(content="你好，我是" + state["username"])]
    }


def llm_node(state: OverAllState) -> OverAllState:
    res = model.invoke(state["messages"])
    return {
        "messages": res,
        "output": res.content
    }

# 3. 构建图
builder = StateGraph(state_schema=OverAllState)
builder.add_node("node_a", node_a)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "llm_node")
builder.add_edge("llm_node", END)

# 4 编译图
graph = builder.compile()

# 5 运行图
result = graph.invoke(OverAllState({"username": "张三"}))
print(result)

