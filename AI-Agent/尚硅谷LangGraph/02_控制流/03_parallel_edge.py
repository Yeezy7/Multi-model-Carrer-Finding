from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models.tongyi import ChatTongyi
from typing import TypedDict

model = ChatTongyi(model="qwen3-max")


class OverAllState(TypedDict):
    topic: str
    poem: str
    joke: str


# 2. 定义节点
def node_a(state: OverAllState) -> OverAllState:
    poem = model.invoke(input=f"写一首关于{state['topic']}主题的诗").content
    return {
        "poem": poem
    }


def node_b(state: OverAllState) -> OverAllState:
    joke = model.invoke(input=f"写一个关于{state['topic']}的笑话").content
    return {
        "joke": joke
    }


# 3, 构建图
builder = StateGraph(state_schema=OverAllState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

# 4. 编译图
graph = builder.compile()
res = graph.invoke({"topic": "猫咪"})
print(res)
