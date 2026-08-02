"""图节点覆盖状态 Overwrite"""
from time import sleep
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


# 1. 定义全局状态
class OverAllState(TypedDict) :
    """全局状态"""
    log: Annotated[list[str], add]  # 规约的方式是add，追加合并
    cur_id: Annotated[str, add]     # 如果出现并行节点，同时更新状态，往下游节点传递的时候，必须加上reducer函数


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
        "log": ["node_2 运行完毕"],   # 覆盖之前的内容
        "cur_id": "node_2"
    }


def node_3(state: OverAllState) -> OverAllState:
    sleep(1)
    for k, v in state.items():
        print(f"node_3: k: {k} v: {v}")
    print("睡眠1秒")
    return {
        "log": ["node_3 运行完毕"],
        "cur_id": "node_3"
    }


def node_4(state: OverAllState) -> OverAllState:
    sleep(2)
    for k, v in state.items():
        print(f"node_4: k: {k} v: {v}")
    print("睡眠2秒")
    return {
        "log": ["node_4 运行完毕"],
    }

builder = StateGraph(state_schema=OverAllState)
# 添加节点
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_node("node_4", node_4)

# 添加边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", "node_4")
builder.add_edge("node_3", "node_4")
builder.add_edge("node_4", END)

# 编译图
graph = builder.compile()
result = graph.invoke({"log": [], "cur_id": "start"})
print(f"="*20, "运行结果", "="*20)
print(result)
