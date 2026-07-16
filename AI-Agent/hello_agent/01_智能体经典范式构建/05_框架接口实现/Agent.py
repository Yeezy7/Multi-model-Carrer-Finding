"""
Agent 类是整个框架的顶层抽象。它定义了一个智能体应该具备的通用行为和属性，但并不关心具体的实现方式。
我们通过 Python 的 abc (Abstract Base Classes) 模块来实现它，
这强制所有具体的智能体实现（如后续章节的 SimpleAgent, ReActAgent 等）都必须遵循同一个“接口”。
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from Message import Message
from Config import Config
from LLM import HelloAgentLLM


class Agent(ABC):
    """Agent 基类"""
    def __init__(self,
                 name: str,
                 llm: HelloAgentLLM,
                 system_prompt: str,
                 config: Optional[Config] = None
    ):
        super().__init__()
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history : list[Message] = []
        
    @abstractmethod # 抽象方法
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass  # 具体实现由子类完成
    
    def add_message(self, message: Message):
        """添加消息到历史记录中"""
        self._history.append(message)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    