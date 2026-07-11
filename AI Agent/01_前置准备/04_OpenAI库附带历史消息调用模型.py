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
        {"role": "system", "content": "你是一个AI助理，回答很简洁。"},
        {"role": "user", "content": "小明有两条宠物狗"},
        {"role": "assistant", "content": "好的，我知道了。"},
        {"role": "user", "content": "小红有3只宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "总共有几只宠物呢？"}
    ],
    stream=True
)

# 3. 处理结果
for chunk in response:
    print(chunk.choices[0].delta.content, 
          end="",    # 每一段之间以空格分隔
          flush=True # 立刻刷新缓冲区
        ) 