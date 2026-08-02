from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Sequence
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.types import Send

model = ChatTongyi(model="qwen3-max")

CONTENT_TYPES = ["poem", "joke", "ci_poem"]

# 1. 定义状态
# 1.1 全局状态
class OverAllState(TypedDict):
    topic: str
    poem: str
    ci_poem: str
    joke: str


# 1.2 私有状态
class WorkState(TypedDict):
    content_type: Literal["poem", "joke", "ci_poem"]
    prompt: str

# 1.3 输入状态
class InputState(TypedDict):
    topic: str


# 1.4 输出状态
class OutputState(TypedDict):
    poem: str
    ci_poem: str
    joke: str


# 2. 定义节点
def worker_node(state: WorkState) -> WorkState:
    """工作节点"""
    content_type = state["content_type"]
    prompt = state["prompt"]
    content = model.invoke(input=prompt).content
    return {
        content_type: content
    }


# 3. 定义动态分支的路由
def route(state: InputState) -> Sequence[Send]:
    """动态路由"""
    router_prompt = "请生成关于{} 的 {}"
    englist2Chinese = {
        "poem": "七言绝句",
        "joke": "笑话",
        "ci_poem": "中文词",
    }
    topic = state["topic"]
    return [
        Send(
            "worker_node",
            {
                "content_type": content_type,
                "prompt": router_prompt.format(topic, englist2Chinese[content_type]),
            }
        )
        for content_type in CONTENT_TYPES
    ]


builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)

builder.add_node("worker_node", worker_node)
builder.add_conditional_edges(START, route, path_map=["worker_node"])
builder.add_edge("worker_node", END)
graph = builder.compile()

res = graph.invoke({"topic": "石头"})
print(res)