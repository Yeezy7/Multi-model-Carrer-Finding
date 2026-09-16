"""使用 LoRA-R 输出 Query-Frame 相关性分数 S_q(t)。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from retrieval_data import load_manifest, retrieval_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=512 * 32 * 32)
    parser.add_argument("--validate-only", action="store_true", help="只校验 manifest、标签 token 和视觉输入，不加载模型")
    return parser.parse_args()


def build_inference_batch(processor: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    instances = []
    for record in records:
        with Image.open(record["image"]) as opened:
            image = opened.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": retrieval_prompt(str(record["instruction"]))},
                ],
            }
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        instances.append({key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)})
    input_ids = [instance["input_ids"][0] for instance in instances]
    padded_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    batch: Dict[str, torch.Tensor] = {
        "input_ids": padded_ids,
        "attention_mask": padded_ids.ne(processor.tokenizer.pad_token_id),
    }
    for key in ("pixel_values", "image_grid_thw"):
        values = [instance[key] for instance in instances if key in instance]
        if values:
            batch[key] = torch.cat(values, dim=0)
    return batch


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"标签 {text!r} 不是单 token，当前分数器要求 0/1 均为单 token: {ids}")
    return int(ids[0])


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if not args.model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    if not args.validate_only and not args.adapter.is_dir():
        raise FileNotFoundError(f"LoRA adapter 不存在: {args.adapter}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor_source = args.adapter if args.adapter.is_dir() else args.model
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    positive_id = single_token_id(processor.tokenizer, "1")
    negative_id = single_token_id(processor.tokenizer, "0")
    if args.validate_only:
        batch = build_inference_batch(processor, records[: min(args.batch_size, len(records))])
        print(json.dumps({
            "validated_samples": len(records),
            "token_ids": {"positive": positive_id, "negative": negative_id},
            "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        }, ensure_ascii=False, indent=2))
        return
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch = build_inference_batch(processor, batch_records)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch, logits_to_keep=1)
            logits = outputs.logits[:, -1, :]
            margin = logits[:, positive_id] - logits[:, negative_id]
            scores = torch.sigmoid(margin).float().cpu().tolist()
            for record, score in zip(batch_records, scores):
                result = dict(record)
                result["score"] = float(score)
                result["prediction"] = int(score >= 0.5)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(records), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
