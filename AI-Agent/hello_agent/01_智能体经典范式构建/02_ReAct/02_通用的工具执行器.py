from typing import Dict, Any


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

if __name__ == '__main__':
    executor = ToolExecutor()
    
    # 注册一个示例工具
    executor.registerTool(
        name="示例工具",
        description="这是一个示例工具，用于演示注册和执行。",
        function=lambda x: f"你输入了: {x}"
    )
    
    # 获取并执行工具
    tool_func = executor.getTool("示例工具")
    if tool_func:
        result = tool_func("Hello, World!")
        print(result)
    
    # 打印所有可用工具
    print("\n可用工具列表:")
    print(executor.getAvailableTools())
    