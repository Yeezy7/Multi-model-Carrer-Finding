
"""
接收一个 LLM 客户端，初始化内部的规划器和执行器，并提供一个简单的 run 方法来启动整个流程。
"""
from tools.Executor import Executor
from tools.Planner import Planner
from llm_client import HelloAgentLLM


class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentLLM):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)
        
    def run(self, question: str) -> str:
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        
        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return 
        
        # 2. 调用执行器逐步执行计划
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

if __name__ == '__main__':
    llm_client = HelloAgentLLM()
    agent = PlanAndSolveAgent(llm_client)
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")