from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class InputState(TypedDict):
    """输入状态"""
    username: str

class OutputState(TypedDict):
    """输出状态"""
    graph_output: str

class OverAllState(TypedDict):
    """全局状态"""
    username: str
    graph_ooutput: str
    nickname: str

class PrivateState(TypedDict):
    """私有状态"""
    greeting: str

# 第一个节点：对接start -> InputState 修改的状态内容在全局状态中 -> OverAllState
def node_1(state: InputState) -> OverAllState:
    """向全局状态添加username"""
    return {
        "nickname": "Dear" + state["username"]
    }

# 第二个节点：对接node1 -> OverAllState 使用的参数在全局状态中，修改的参数在私有状态中 -> PrivateState
def node_2(state: OverAllState) -> PrivateState:
    # 向私有状态添加greeting
    return {
        "greeting": "Hello, " + state["nickname"]
    }

# 第三个节点：对接node2 —> PrivateState 使用的参数在私有状态中，修改的参数在输出状态中 -> OutputState
def node_3(state: PrivateState) -> OutputState:
    """向输出状态添加graph_output"""
    return {
        "graph_output": state["greeting"] + " 很高兴认识你！"
    }


# 构建状态图
# 定义图的时候 加载全局状态 输入状态 输出状态
builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)

# 添加节点
# 添加节点的时候  加载私有状态
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 添加边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
result = graph.invoke({"username": "autssdsd"})
print(result)