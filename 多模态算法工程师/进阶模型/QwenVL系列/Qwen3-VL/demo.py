import argparse
import os
from typing import Optional

from PIL import Image

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少依赖: pip install transformers peft accelerate pillow" 
    ) from exc


def build_prompt(question: str) -> str:
    return (
        "请根据图片内容进行回答。输出简洁准确，优先直接给出结论。\n"
        f"问题：{question}"
    )


def run_inference(model_name: str, image_path: str, question: str) -> str:
    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL 可视化问答 demo")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument(
        "--question",
        type=str,
        default="请总结这张图中最关键的信息。",
        help="问题内容",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN3_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="模型名或本地路径",
    )
    args = parser.parse_args()

    result = run_inference(args.model, args.image, args.question)
    print("\n=== 输出 ===\n")
    print(result)
