from typing import Callable, Any
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from utils.logger_handler import logger
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from utils.prompt_loader import load_report_prompt, load_system_prompt


@wrap_tool_call
def monitor_tool(  
    request: ToolCallRequest,    # 工具调用请求
    handler: Callable[[ToolCallRequest], ToolMessage | Command],    # 工具调用处理器
) -> ToolMessage | Command:  
    """监控工具调用"""
    logger.info(f"[tool monitor] 工具调用请求: {request.tool_call['name']}")
    logger.info(f"[tool monitor] 工具调用参数: {request.tool_call['args']}")
    try:
        result =  handler(request)
        logger.info(f"[tool monitor] 工具{request.tool_call['name']}调用成功: {result}")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        
        return result
    except Exception as e:
        logger.error(f"[tool monitor] 工具{request.tool_call['name']}调用时异常: {str(e)}")
        raise e


@before_model
def log_before_model(
    state: AgentState,  # 整个Agent智能体中的状态记录
    runtime: Runtime    # 记录整个执行过程中的上下文信息

):   # 
    """在模型执行前输出日志"""
    logger.info(f"[log_before_model] 即将调用模型，带有{len(state['messages'])}条消息记录。")
    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")    
    return None

@dynamic_prompt             # 在每一次生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest): 
    """根据上下文动态切换提示词"""
    is_report = request.runtime.context.get("report", False)    # 获取上下文中是否为报告生成的标记
    if is_report:
        return load_report_prompt()
    return load_system_prompt()  # 默认返回系统提示词



def list_middleware() -> list[Callable[..., Any]]:
    """列出所有中间件"""
    return [
        monitor_tool,
        log_before_model,
        report_prompt_switch,
    ]