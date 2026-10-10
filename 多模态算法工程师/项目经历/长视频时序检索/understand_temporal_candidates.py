"""使用 Qwen3-VL 对候选时间窗口进行多帧片段级理解。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=512 * 32 * 32)
    return parser.parse_args()


def load_frame_index(frame_root: Path, video_id: str) -> list[tuple[float, Path]]:
    paths = sorted((frame_root / video_id).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"候选视频没有帧: {video_id}")
    return [(float(path.stem), path) for path in paths]


def choose_frames(
    frame_index: list[tuple[float, Path]], start: float, end: float, num_frames: int
) -> list[tuple[float, Path]]:
    inside = [(timestamp, path) for timestamp, path in frame_index if start <= timestamp <= end]
    if not inside:
        inside = frame_index
    if len(inside) <= num_frames:
        return inside
    targets = [
        start + (end - start) * index / (num_frames - 1)
        for index in range(num_frames)
    ] if num_frames > 1 else [(start + end) / 2.0]
    selected: list[tuple[float, Path]] = []
    used: set[Path] = set()
    for target in targets:
        timestamp, path = min(inside, key=lambda item: abs(item[0] - target))
        if path not in used:
            selected.append((timestamp, path))
            used.add(path)
    return selected


def build_prompt(query: str, start: float, end: float) -> str:
    return (
        "你正在进行长视频时序检索的候选片段理解。\n"
        f"Query：{query}\n"
        f"候选片段时间：{start:.1f}s - {end:.1f}s。\n"
        "下面是该片段按时间顺序抽取的多帧画面。\n"
        "请判断这些画面是否包含 Query 描述的事件。只输出 YES 或 NO，不要输出解释。"
    )


def parse_binary_answer(text: str) -> int | None:
    normalized = text.strip().upper()
    if normalized.startswith("YES") or normalized.startswith("是"):
        return 1
    if normalized.startswith("NO") or normalized.startswith("否"):
        return 0
    return None


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.num_frames <= 0:
        raise ValueError("top-k 和 num-frames 必须大于 0")
    if not args.model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    if args.adapter is not None and not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter 目录不存在: {args.adapter}")

    query_rows = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_rows: list[dict[str, Any]] = []
    for query in query_rows:
        for candidate in query.get("candidates", [])[: args.top_k]:
            candidate_rows.append(
                {
                    "query_id": query["query_id"],
                    "video_id": query["video_id"],
                    "instruction": query["instruction"],
                    "gt_segments": query["gt_segments"],
                    **candidate,
                }
            )
    if args.max_candidates is not None:
        candidate_rows = candidate_rows[: args.max_candidates]
    if not candidate_rows:
        raise ValueError("候选片段为空")

    processor_source = args.adapter if args.adapter is not None else args.model
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    frame_cache: dict[str, list[tuple[float, Path]]] = {}
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_rows, 1):
        video_id = str(candidate["video_id"])
        frame_cache.setdefault(video_id, load_frame_index(args.frame_root, video_id))
        selected_frames = choose_frames(
            frame_cache[video_id],
            float(candidate["start"]),
            float(candidate["end"]),
            args.num_frames,
        )
        images = [Image.open(path).convert("RGB") for _, path in selected_frames]
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {
                        "type": "text",
                        "text": build_prompt(
                            str(candidate["instruction"]),
                            float(candidate["start"]),
                            float(candidate["end"]),
                        ),
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_only = generated[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        for image in images:
            image.close()
        result = dict(candidate)
        result.update(
            {
                "frame_timestamps": [round(timestamp, 3) for timestamp, _ in selected_frames],
                "answer": answer,
                "prediction": parse_binary_answer(answer),
                "ground_truth_positive_iou03": float(candidate["best_temporal_iou"]) >= 0.3,
            }
        )
        results.append(result)
        if index % 10 == 0 or index == len(candidate_rows):
            print(f"processed {index}/{len(candidate_rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    valid = [result for result in results if result["prediction"] is not None]
    tp = sum(result["prediction"] == 1 and result["ground_truth_positive_iou03"] for result in valid)
    fp = sum(result["prediction"] == 1 and not result["ground_truth_positive_iou03"] for result in valid)
    fn = sum(result["prediction"] == 0 and result["ground_truth_positive_iou03"] for result in valid)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "candidates": len(results),
        "queries": len({result["query_id"] for result in results}),
        "valid_binary_answers": len(valid),
        "parse_failure": len(results) - len(valid),
        "precision_iou03": precision,
        "recall_iou03": recall,
        "output": str(args.output),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
