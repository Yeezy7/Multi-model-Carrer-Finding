"""
Step 5: 测试微调后的 Qwen-VL（属性抽取 + 一致性判断）。

用法：
    python test_finetuned.py --image ../datasets/products10k_images/img_00000.jpg
    python test_finetuned.py --image xxx.jpg --lora ./output_lora
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 本地基座模型路径
MODEL_PATH = ("pretrained_ckpt/models--Qwen--Qwen-VL-Chat/snapshots/"
              "f57cfbd358cb56b710d963669ad1bcfb44cdcdd8")

EXTRACT_PROMPT = (
    "请抽取这张商品图片的属性信息，用 JSON 格式输出："
    '包含 "category"（类别）、"color"（颜色）、"material"（材质）、'
    '"text"（图中可见文字）、"features"（显著特征）字段，只输出 JSON。'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    parser.add_argument("--lora", type=str, default="./output_lora",
                        help="LoRA 权重目录（微调后）")
    parser.add_argument("--desc", type=str, default=None,
                        help="一致性判断：一段商品描述")
    return parser.parse_args()


def main():
    args = parse_args()

    print("加载基座模型 + LoRA 权重 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cuda", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    model = PeftModel.from_pretrained(model, args.lora)
    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )

    # 1. 属性抽取
    print("\n===== 任务1: 属性抽取 =====")
    query = tokenizer.from_list_format([
        {"image": args.image},
        {"text": EXTRACT_PROMPT},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    print(f"图片: {args.image}")
    print(f"回答: {response}")

    # 2. 一致性判断（可选）
    if args.desc:
        print("\n===== 任务2: 一致性判断 =====")
        q = (f"下面这段商品描述与图片内容一致吗？描述：\"{args.desc}\" "
             "请回答一致或不一致，并简要说明理由。")
        query = tokenizer.from_list_format([
            {"image": args.image},
            {"text": q},
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
        print(f"描述: {args.desc}")
        print(f"回答: {response}")


if __name__ == "__main__":
    main()
