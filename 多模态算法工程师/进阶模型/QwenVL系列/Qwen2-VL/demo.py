import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B", trust_remote_code=True, cache_dir="./pretrained_ckpt")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-VL-7B", device_map="cuda", trust_remote_code=True, cache_dir="./pretrained_ckpt").eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2-VL-7B", trust_remote_code=True)
