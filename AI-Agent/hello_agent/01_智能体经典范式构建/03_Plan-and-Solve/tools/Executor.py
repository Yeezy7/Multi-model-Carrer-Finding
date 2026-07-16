from llm_client import HelloAgentLLM


# 原始问题： 确保模型始终了解最终目标。
# 完整计划： 让模型了解当前步骤在整个任务中的位置。
# 历史步骤与结果： 提供至今为止已经完成的工作，作为当前步骤的直接输入。
# 当前步骤： 明确指示模型现在需要解决哪一个具体任务。


EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""


class Executor:
    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client
        
    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = " " # 用于存储历史步骤和结果的字符串
        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",  # 如果是第一步，则历史为空
                current_step=step
            )
            message = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=message) or ""

            # 更新历史记录，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            
            print(f"✅ 步骤 {i+1} 完成，结果: {response_text}")
        
        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer


if __name__ == '__main__':
    executor = Executor(llm_client=HelloAgentLLM())
    executor.execute(
        question="请帮我制定一个计划，了解华为最新手机型号及其主要卖点。",
        plan=["访问华为官方网站的智能手机产品页面", "查找最新发布的手机型号（通常标记为'新品'或按发布时间排序）", "记录该型号的名称和发布日期", "浏览该型号的详细规格页面，提取主要卖点（如摄像头性能、电池续航、芯片型号、屏幕技术等）", "查阅权威科技媒体或评测网站对该型号的评测，验证并补充其核心优势", "整理汇总所有信息，形成一份包含型号名称和主要卖点的简明清单"])