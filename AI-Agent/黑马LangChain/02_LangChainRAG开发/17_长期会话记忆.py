import os, json
from typing import Sequence
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
# message_to_dict: 单个消息对象(BaseMessage类实例) -> 字典
# messages_from_dict: [字典、字典...] -> [消息、消息...]   
from langchain_core.chat_history import BaseChatMessageHistory
# AIMessage、HumanMessage、SystemMessage、ChatMessage等类都是BaseMessage的子类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.history import RunnableWithMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id          # 会话id
        self.storage_path = storage_path      # 不同会话id的存储文件，所在的文件夹路径
        
        self.file_path = os.path.join(self.storage_path, self.session_id)  # 存储文件路径
        
        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
    def add_messages(self, messages: Sequence[BaseMessage]):
        """添加消息到历史记录中"""
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)       # 新的和已有的消息合并
        
        # 将数据同步写入到本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典，以json格式写入文件
        # message_to_dict: 单个消息对象(BaseMessage类实例) -> 字典
        
        # new_messages = []
        # for message in all_messages:
        #     message_dict = message_to_dict(message) 
        #     new_messages.append(message_dict)
        new_messages = [message_to_dict(message) for message in all_messages]
        
        # 将字典列表写入文件
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(new_messages, f)
    
    @property  # 将messages方法变成成员属性
    def messages(self) -> list[BaseMessage]:
        """获取历史消息"""
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)
                # messages_from_dict: [字典、字典...] -> [消息、消息...]   
                return messages_from_dict(messages_data)  # [字典、字典...] -> [消息、消息...]
        except FileNotFoundError:
            return [] # 如果文件不存在，返回空列表
        
    def clear(self) -> None:
        """清空历史消息"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)  # 写入空列表，表示清空历史消息


model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template(
#     "你需要根据会话历史回应用户问题。对话历史：{chat_history}，用户问题：{input}，请给出简明扼要的回答。"
# )

prompt = ChatPromptTemplate(
    messages=[
        ("system", "你需要根据会话历史回应用户问题。对话记录："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题：{input}")
    ]
)

str_parser = StrOutputParser()

def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

base_chain = prompt | print_prompt | model | str_parser


store = {}  # key 就是 session_id, value 就是 InMemoryChatMessageHistory 类对象
# 实现通过会话id获取InMemoryChatMessageHistory类对象的函数
def get_history(session_id):
    return FileChatMessageHistory(session_id=session_id, storage_path="./chat_history")

# 创建一个新的链，对原有链增强功能：自动附加历史消息
conversation_chain = RunnableWithMessageHistory(
    runnable=base_chain, # 被增强的原有chain
    get_session_history=get_history, # 通过会话id获取InMemoryChatMessageHistory类对象
    input_messages_key="input",  # 表示用户输入在模板中的占位符
    history_messages_key="chat_history",  # 表示历史消息在模板中的占位符
)


if __name__ == "__main__":
    # 固定格式，添加LangChain的配置，为当前程序配置所属的session_id
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    # res = conversation_chain.invoke({"input": "小明有两个猫"}, config=session_config)
    # print("第一次执行：", res)
    # res = conversation_chain.invoke({"input": "小刚有一只狗"}, config=session_config)
    # print("第二次执行：", res)
    res = conversation_chain.invoke({"input": "一共有几只宠物"}, config=session_config)
    print("第三次执行：", res)
