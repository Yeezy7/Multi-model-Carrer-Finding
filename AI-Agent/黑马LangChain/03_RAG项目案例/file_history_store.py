from langchain_core.chat_history import BaseChatMessageHistory
import config_data as config
import json
from langchain_core.messages import messages_from_dict, message_to_dict, BaseMessage
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import Sequence

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = config.storage_path
        self.file_path = f"{self.storage_path}/{self.session_id}"
        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]):
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)       # 新的和已有的消息合并
        
        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(new_messages, f)
    
    
    @property  # 将messages方法变成成员属性
    def messages(self) -> list[BaseMessage]:
        """获取历史消息"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []  # 如果文件不存在，返回空列表
    
    def clear(self) -> None:
        """清空历史消息"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)  # 写入空列表，表示清空历史消息
    
    
if __name__ == "__main__":
    
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    
    model = ChatTongyi(model="qwen3-max")
    prompt = ChatPromptTemplate(
        [
            ("system", "你需要根据会话历史回应用户问题。对话历史：{chat_history}，用户问题：{input}，请给出简明扼要的回答。"),
            ("user", "{input}")
        ]
    )
    
    def print_prompt(prompt):
        print("="*10 + "提示词" + "="*10)
        print(prompt.to_string())
        print("="*20)
        return prompt   
    
    base_chain = prompt | print_prompt | model | StrOutputParser()
    chain = RunnableWithMessageHistory(
        runnable=base_chain,
        get_session_history=lambda session_id: FileChatMessageHistory(session_id, config.storage_path),
        input_messages_key="input",
        output_messages_key="output"
    )