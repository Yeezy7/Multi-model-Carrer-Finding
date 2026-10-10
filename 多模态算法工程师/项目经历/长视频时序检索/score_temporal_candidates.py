"""使用 Qwen3-VL 的 0/1 logit 对多帧候选片段打分并重排。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from temporal_understanding_data import understanding_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10, help="每个 Query 参与重排的原始候选数")
    parser.add_argument("--num-frames", type=int, default=4)
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
    targets = (
        [start + (end - start) * index / (num_frames - 1) for index in range(num_frames)]
        if num_frames > 1
        else [(start + end) / 2.0]
    )
    selected: list[tuple[float, Path]] = []
    used: set[Path] = set()
    for target in targets:
        timestamp, path = min(inside, key=lambda item: abs(item[0] - target))
        if path not in used:
            selected.append((timestamp, path))
            used.add(path)
    while len(selected) < num_frames:
        selected.append(selected[-1])
    return selected


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"标签 {text!r} 不是单 token: {ids}")
    return int(ids[0])


def build_rows(path: Path, top_k: int) -> list[dict[str, Any]]:
    groups = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for query in groups:
        candidates = sorted(query.get("candidates", []), key=lambda item: int(item["rank"]))[:top_k]
        for candidate in candidates:
            rows.append(
                {
                    "query_id": query["query_id"],
                    "video_id": query["video_id"],
                    "instruction": query["instruction"],
                    "gt_segments": query.get("gt_segments", []),
                    **candidate,
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]], order_key: str, top_ks: list[int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    query_metrics: dict[str, dict[str, float]] = {}
    for query_id, query_rows in grouped.items():
        ordered = sorted(query_rows, key=lambda row: row[order_key])
        metrics: dict[str, float] = {}
        for k in top_ks:
            selected = ordered[:k]
            best = max((float(row["best_temporal_iou"]) for row in selected), default=0.0)
            metrics[f"mean_best_tiou@{k}"] = best
            metrics[f"hit@{k}"] = float(best > 0.0)
            for threshold in (0.1, 0.3, 0.5):
                metrics[f"recall@{k}_tiou{threshold}"] = float(best >= threshold)
        query_metrics[query_id] = metrics

    def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: sum(item[key] for item in items) / len(items)
            for key in items[0]
        } if items else {}

    by_video: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in grouped.values():
        by_video[str(row[0]["video_id"])].append(query_metrics[str(row[0]["query_id"])])
    return {
        "queries": len(query_metrics),
        "videos": len(by_video),
        "query_macro": mean_dict(list(query_metrics.values())),
        "video_macro": mean_dict([mean_dict(items) for items in by_video.values()]),
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.num_frames <= 0:
        raise ValueError("top-k 和 num-frames 必须大于 0")
    if not args.model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    if args.adapter is not None and not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter 目录不存在: {args.adapter}")

    rows = build_rows(args.candidates, args.top_k)
    if args.max_candidates is not None:
        rows = rows[: args.max_candidates]
    if not rows:
        raise ValueError("候选片段为空")

    processor_source = args.adapter if args.adapter is not None else args.model
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    positive_id = single_token_id(processor.tokenizer, "1")
    negative_id = single_token_id(processor.tokenizer, "0")
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
    for index, row in enumerate(rows, 1):
        video_id = str(row["video_id"])
        frame_cache.setdefault(video_id, load_frame_index(args.frame_root, video_id))
        selected = choose_frames(
            frame_cache[video_id], float(row["start"]), float(row["end"]), args.num_frames
        )
        images = []
        try:
            for _, path in selected:
                with Image.open(path) as opened:
                    images.append(opened.convert("RGB"))
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": understanding_prompt(str(row["instruction"]))},
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
            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode():
                outputs = model(**inputs, logits_to_keep=1)
            logits = outputs.logits[:, -1, :]
            margin = (logits[:, positive_id] - logits[:, negative_id]).item()
            score = float(torch.sigmoid(torch.tensor(margin)).item())
        finally:
            for image in images:
                image.close()
        result = dict(row)
        result.update(
            {
                "frame_timestamps": [round(timestamp, 3) for timestamp, _ in selected],
                "lora_u_margin": margin,
                "lora_u_score": score,
                "ground_truth_positive_iou03": float(row["best_temporal_iou"]) >= 0.3,
            }
        )
        results.append(result)
        if index % 10 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["query_id"])].append(result)
    for query_rows in grouped.values():
        original = sorted(query_rows, key=lambda row: int(row["rank"]))
        reranked = sorted(query_rows, key=lambda row: (-float(row["lora_u_score"]), int(row["rank"])))
        original_rank = {id(row): rank for rank, row in enumerate(original, 1)}
        rerank_rank = {id(row): rank for rank, row in enumerate(reranked, 1)}
        for row in query_rows:
            row["original_rank"] = original_rank[id(row)]
            row["lora_u_rank"] = rerank_rank[id(row)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    top_ks = sorted({1, min(5, args.top_k), args.top_k})
    summary = {
        "candidates": len(results),
        "queries": len(grouped),
        "top_k_scored": args.top_k,
        "original_rank": aggregate(results, "original_rank", top_ks),
        "lora_u_rerank": aggregate(results, "lora_u_rank", top_ks),
        "output": str(args.output),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
