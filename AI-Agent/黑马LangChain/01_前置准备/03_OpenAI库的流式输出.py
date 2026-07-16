from openai import OpenAI
import os

# 1. 获取client对象，OpenAI类对象
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2. 调用模型
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个Python编程专家，并且话很多。"},
        {"role": "assistant", "content": "好的，我是编程专家，并且话很多，你要问什么？"},
        {"role": "user", "content": "请帮我写一个Python函数，输入一个字符串，返回该字符串的长度。"}
    ],
    stream=True
)

# 3. 处理结果
for chunk in response:
    print(chunk.choices[0].delta.content, 
          end="",    # 每一段之间以空格分隔
          flush=True # 立刻刷新缓冲区
        ) 