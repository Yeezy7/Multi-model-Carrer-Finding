import os
import sys
from typing import Dict, Any, List
from serpapi import SerpApiClient
from dotenv import load_dotenv
from openai import OpenAI
import re

# 加载环境变量
load_dotenv("AI-Agent/hello_agent/.env")


# ReAct 提示词模板

# 角色定义： “你是一个有能力调用外部工具的智能助手”，设定了LLM的角色。
# 工具清单 ({tools})： 告知LLM它有哪些可用的“手脚”。
# 格式规约 (Thought/Action)： 这是最重要的部分，它强制LLM的输出具有结构性，使我们能通过代码精确解析其意图。
# 动态上下文 ({question}/{history})： 将用户的原始问题和不断累积的交互历史注入，让LLM基于完整的上下文进行决策。
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


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


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("请确保已设置SERPAPI_API_KEY环境变量。")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }
        
        clinet = SerpApiClient(params)
        results = clinet.get_dict()
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')} "
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"
    
    except Exception as e:
        return f"❌ 搜索时发生错误: {e}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        
    def registerTool(self, name: str, description: str, function: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "function": function
        }
        print(f"✅ 工具 '{name}' 已注册。描述: {description}")
        
    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("function", None)

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


class ReActAgent:
    def __init__(self, llm_client: HelloAgentLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
    
    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        负责从LLM的完整响应中分离出Thought和Action两个主要部分。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:(.*?)(?=Action:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action
    
    def _parse_action(self, action_text: str):
        """
        解析Action字符串，提取工具名称和输入。
        负责进一步解析Action字符串，例如从 Search[华为最新手机] 中提取出工具名 Search 和工具输入 华为最新手机。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
        
    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")
            
            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            
            # 2. 调用LLM进行思考
            messages = [
                {"role": "user", "content": prompt}
            ]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("❌ LLM 未返回有效响应。")
                break
            
            # 3. 解析、执行、整合
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break
            
            # 4. 执行action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue
            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = None
            if not tool_function:
                print(f"❌ 未找到名为 '{tool_name}' 的工具。")
            else:
                observation = tool_function(tool_input) # 调用真实工具
                # 观测结果的整合
                print(f"👀 观察: {observation}")
                # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        # 循环结束
        print("已达到最大步数，流程终止。")
        return None


if __name__ == '__main__':
    
    tool_executor=ToolExecutor()
    tool_executor.registerTool(
        "Search", 
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。", 
        search
    )
    
    reactagent = ReActAgent(llm_client=HelloAgentLLM(), tool_executor=tool_executor)
    reactagent.run("华为最新手机型号及主要卖点")