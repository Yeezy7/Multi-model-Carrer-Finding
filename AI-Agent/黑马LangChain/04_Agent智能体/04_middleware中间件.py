from langchain.agents import create_agent, AgentState
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langchain.agents.middleware import (
    before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call
)
from langgraph.runtime import Runtime

@tool(description="查询天气，传入城市字符串，返回字符串城市天气信息")
def get_weather(city: str) -> str:
    return f"{city}的天气是晴天"

"""
1. agent执行前
2. agent执行后
3. model执行前
4. model执行后
5. 工具执行中
6. 模型执行中
"""

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """在agent执行前会调用这个函数，并传入state、runtime两个参数"""
    print(f"[before agent] agent启动，并附带{len(state['messages'])}条消息")

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    """在agent执行后会调用这个函数，并传入state、runtime两个参数"""
    print(f"[after agent] agent执行完毕，并附带{len(state['messages'])}条消息")

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """在model执行前会调用这个函数，并传入state、runtime两个参数"""
    print(f"[before model] model即将调用，并附带{len(state['messages'])}条消息") 
    
@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    """在model执行后会调用这个函数，并传入state、runtime两个参数"""
    print(f"[after model] model执行完毕，并附带{len(state['messages'])}条消息")
    
    
@wrap_model_call
def model_call_hook(request, handler):
    """在model调用前会调用这个函数，并传入request、handler两个参数"""
    print(f"模型被调用啦，请求参数：{request}")
    return handler(request)

@wrap_tool_call
def  monitor_tool(request, handler):
    """在工具调用前会调用这个函数，并传入request、handler两个参数"""
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数：{request.tool_call['args']}")
    return handler(request)


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weather],
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool],
)

for chunk in agent.stream(
        {
            "messages": [{"role": "user", "content": "查询郑州天气"}]
        }, stream_mode="values"
    ):
    latest_message = chunk['messages'][-1]
    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)
    
    
    try:
        if latest_message.tool_calls:
            print(f"工具调用：{[tc['name'] for tc in latest_message.tool_calls]}")
    except AttributeError as e:
        pass