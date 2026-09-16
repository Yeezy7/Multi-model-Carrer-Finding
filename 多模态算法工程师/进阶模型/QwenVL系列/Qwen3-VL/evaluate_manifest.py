"""Evaluate a base model or PEFT adapter on unified manifest records, including KIE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from kie_metrics import field_counts, micro_f1, parse_json_output
from multitask_data import load_manifest, task_prompt
from vqa_metrics import anls_score, chartqa_relaxed_accuracy


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("无法定位仓库根目录")


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "pre-trained-model" / "Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=1024 * 32 * 32)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def infer(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    record: Dict[str, Any],
    max_new_tokens: int,
) -> str:
    with Image.open(record["image"]) as opened:
        image = opened.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": task_prompt(record)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if args.tasks:
        selected = {task.lower() for task in args.tasks}
        records = [record for record in records if record["task"] in selected]
    if args.max_samples:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("筛选后没有评测样本")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "load_in_4bit": args.load_in_4bit,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "tasks": sorted(args.tasks),
        "max_samples": args.max_samples,
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(f"输出目录已绑定其他评测协议: {protocol_path}")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    prediction_path = args.output_dir / "predictions.jsonl"
    completed: Dict[str, Dict[str, Any]] = {}
    if prediction_path.exists():
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("status") == "ok":
                    completed[str(record["id"])] = record

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with prediction_path.open("a", encoding="utf-8") as output:
        for index, record in enumerate(records, 1):
            if record["id"] in completed:
                continue
            started = time.perf_counter()
            try:
                prediction = infer(model, processor, record, args.max_new_tokens)
                result = {
                    "id": record["id"],
                    "task": record["task"],
                    "instruction": record["instruction"],
                    "response": record["response"],
                    "prediction": prediction,
                    "latency_seconds": time.perf_counter() - started,
                    "status": "ok",
                }
            except Exception as error:
                result = {
                    "id": record["id"],
                    "task": record["task"],
                    "instruction": record["instruction"],
                    "response": record["response"],
                    "prediction": "",
                    "latency_seconds": time.perf_counter() - started,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed[record["id"]] = result
            if index == 1 or index % args.log_every == 0 or result["status"] == "error":
                print(f"{index}/{len(records)} {record['id']} {result['status']}", flush=True)

    successful = [record for record in completed.values() if record.get("status") == "ok"]
    task_results: Dict[str, Any] = {}
    for task in sorted({record["task"] for record in successful}):
        task_records = [record for record in successful if record["task"] == task]
        if task == "kie":
            counts = []
            valid_json = 0
            for record in task_records:
                reference = json.loads(record["response"])
                try:
                    prediction = parse_json_output(record["prediction"])
                    valid_json += 1
                except (json.JSONDecodeError, TypeError):
                    prediction = {}
                counts.append(field_counts(prediction, reference))
            task_results[task] = {
                **micro_f1(counts),
                "num_samples": len(task_records),
                "json_valid_rate": valid_json / len(task_records),
            }
        else:
            scores: List[float] = []
            for record in task_records:
                if task == "chartqa":
                    score = chartqa_relaxed_accuracy(record["prediction"], record["response"])
                else:
                    score = anls_score(record["prediction"], [record["response"]])
                scores.append(score)
            task_results[task] = {
                "num_samples": len(scores),
                "score_percent": 100 * sum(scores) / len(scores),
            }
    summary = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "num_runtime_errors": sum(record.get("status") != "ok" for record in completed.values()),
        "tasks": task_results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
