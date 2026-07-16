import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import list_tools
from agent.tools.middleware import list_middleware

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,  # 大模型
            system_prompt=load_system_prompt(), # 系统提示词
            tools=list_tools(),                 # 工具函数
            middleware=list_middleware(),       # 中间件
        )
    
    
    def execute_stream(self, question: str):
        """"""
        input_dict = {
            "messages": [
                {"role" : "user", "content": question},
            ]
        }
        
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
                
if __name__ == "__main__":
    react_agent = ReactAgent()
    question = "给我生成我的使用报告"
    for chunk in react_agent.execute_stream(question):
        print(chunk, end="", flush=True)