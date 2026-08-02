"""图节点读取状态"""

from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


# 1. 定义全局状态
class OverAllState(TypedDict) :
    """全局状态"""
    log: Annotated[list[str], add]  # 规约的方式是add，追加合并
    cur_id: str


# 2. 定义节点
def node_1(state: OverAllState) -> OverAllState:
    for k, v in state.items():
        print(f"node_1: k: {k} v: {v}")
    return {
        "log": ["node_1 运行完毕"],
    }

def node_2(state: OverAllState) -> OverAllState:
    for k, v in state.items():
        print(f"node_2: k: {k} v: {v}")
    return {
        "log": ["node_2 运行完毕"],
    }

def node_3(state: OverAllState) -> OverAllState:
    for k, v in state.items():
        print(f"node_3: k: {k} v: {v}")
    return {
        "log": ["node_3 运行完毕"],
    }

builder = StateGraph(state_schema=OverAllState)
# 添加节点
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 添加边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# 编译图
graph = builder.compile()
result = graph.invoke({"log": [], "cur_id": "start"})
print(result)
