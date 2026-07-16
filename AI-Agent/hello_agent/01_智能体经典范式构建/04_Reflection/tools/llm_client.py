"""LLM客户端类"""

import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("AI-Agent/hello_agent/.env")

class HelloAgentLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or 60
        
        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError("请确保已设置LLM_API_KEY、LLM_MODEL_ID和LLM_BASE_URL环境变量，或在初始化时提供这些参数。")
        self.client = OpenAI(api_key=self.apiKey, base_url=self.baseUrl, timeout=self.timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            # 处理流式响应
            print(f"✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print() # 换行
            return "".join(collected_content)
        
        except Exception as e:
            print(f"❌ 调用大语言模型时出错: {e}")
            return None
    
#  --- 客户端使用示例 ---
if __name__ == '__main__':
    llm = HelloAgentLLM()
    messages = [
        {"role": "system", "content": "你是一个专业的Python代码编写助手，话不多说，直接写代码。"},
        {"role": "user", "content": "写一个快速排序算法"}
    ]
    print("--- 调用LLM ---")
    response = llm.think(messages)
    if response:
        print("\n--- 完整模型响应 ---\n", response)
    else:
        print("LLM调用失败")
