import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

my_cache_dir = "./pretrained_ckpt"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=my_cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, cache_dir=my_cache_dir).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 视觉问答

# 首先，我们使用tokenizer.from_list_format可以对图文混排输入进行分词与处理:
# query = tokenizer.from_list_format([
#     {'image': 'assets/image.png'},
#     {'text': '海报上的电影名字是什么？'},
# ])

# 接下来，我们可以使用model.chat向Qwen-VL-Chat模型提问并获得回复。
# 注意在第一次提问时，对话历史为空，因此我们使用history=None。
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

# # 我们还可以继续向模型发问，例如询问电影的导演是谁。
# # 在后续提问时，对话历史并不为空，我们使用history=history向model.chat传递之前的对话历史：
# query = tokenizer.from_list_format([
#     {'text': '谁执导了这个电影？'},
# ])
# response, history = model.chat(tokenizer, query=query, history=history)
# print(response)

# 文字理解
# query = tokenizer.from_list_format([
#     {'image': 'assets/Hospital.jpeg'},
#     {'text': '根据照片，耳鼻喉科在几楼？'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

# 图表数学推理
query = tokenizer.from_list_format([
    {'image': 'assets/Menu.png'},
    {'text': 'How much would I pay if I want to order two Salmon Burger and three Meat Lover\'s Pizza? Think carefully step by step.'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)