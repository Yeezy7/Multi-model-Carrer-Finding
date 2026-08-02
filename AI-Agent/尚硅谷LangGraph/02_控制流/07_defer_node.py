from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models.tongyi import ChatTongyi
from typing import TypedDict, Literal, Sequence
from loguru import logger


model = ChatTongyi(model="qwen3-max")


class OverAllState(TypedDict):
    topic: str
    poem: str
    joke: str
    ci_poem: str
    content_type: str


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

def audit_node(state: OverAllState) -> OverAllState:
    """延时节点"""
    logger.info(f"任务阶段已经全部执行完毕，诗: {'已生成' if state['poem'] else '未生成'}，"
                f"笑话: {'已生成' if state['joke'] else '未生成'}，")


# 3, 构建图
builder = StateGraph(state_schema=OverAllState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("audit_node", audit_node, defer=True)  # 延时节点

builder.add_edge(START, "node_a")
builder.add_edge(START, "node_b")
builder.add_edge(START, "audit_node")
builder.add_edge("node_a", END)
builder.add_edge("node_b", END)
builder.add_edge("audit_node", END)

# 4. 编译图
graph = builder.compile()
poem_res = graph.invoke({"topic": "猫咪"})
print(poem_res)


joke_res = graph.invoke({"topic": "猫咪"})
print(joke_res)

print(graph.get_graph().print_ascii())
