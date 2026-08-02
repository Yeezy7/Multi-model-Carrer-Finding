from langgraph.graph import StateGraph, START, END
from typing import Annotated
from operator import add
from pydantic import BaseModel

"""pydantic 定义状态"""

class OverAllState(BaseModel):
    logs: Annotated[list[str], add]
    cur_id: str

# 2. 定义节点1
def node_1(state: OverAllState) -> OverAllState:
    """节点1"""
    pre_id = state.cur_id
    return {
        "logs": ["node_1 运行完毕"],
        "cur_id": pre_id + ", node_1"
    }
# 定义节点2
def node_2(state: OverAllState) -> OverAllState:
    """节点2"""
    pre_id = state.cur_id
    return {
        "logs": ["node_2 运行完毕"],
        "cur_id": pre_id + ", node_2"
    }


# 3, 定义边
# 3.1 创建图，获取建造者
builder = StateGraph(state_schema=OverAllState)
# 3.2 添加节点
builder.add_node(node_1)
builder.add_node(node_2)
# 3.3 添加边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# 4 编译图
graph = builder.compile()

# 5 运行图
result = graph.invoke({"cur_id": "start"})
print(result)