"""从候选时间窗口构造 LoRA-U 的多帧候选片段理解清单。"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="build_temporal_candidates.py 输出的 JSONL")
    parser.add_argument("--frame-root", type=Path, required=True, help="按 video_id 分目录保存 JPG 帧")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10, help="每个 Query 最多保留前 K 个候选")
    parser.add_argument("--num-frames", type=int, default=4, help="每个候选窗口均匀抽取的帧数")
    parser.add_argument("--positive-iou", type=float, default=0.3)
    parser.add_argument("--hard-negative-iou", type=float, default=0.1)
    parser.add_argument("--max-negatives-per-query", type=int, default=6)
    parser.add_argument("--include-all-positives", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def frame_index(frame_root: Path, video_id: str) -> tuple[list[float], list[Path]]:
    paths = sorted((frame_root / video_id).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"缺少视频帧目录或 JPG: {frame_root / video_id}")
    pairs = sorted((float(path.stem), path) for path in paths)
    return [item[0] for item in pairs], [item[1] for item in pairs]


def nearest_frame(timestamps: list[float], paths: list[Path], target: float) -> Path:
    index = bisect_left(timestamps, target)
    candidates = []
    if index < len(timestamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(timestamps[item] - target))
    return paths[best]


def select_frame_paths(
    timestamps: list[float],
    paths: list[Path],
    start: float,
    end: float,
    num_frames: int,
) -> list[Path]:
    if end < start:
        raise ValueError(f"候选窗口非法: {start} > {end}")
    if num_frames == 1:
        targets = [(start + end) / 2.0]
    else:
        targets = [start + (end - start) * i / (num_frames - 1) for i in range(num_frames)]
    selected: list[Path] = []
    for target in targets:
        path = nearest_frame(timestamps, paths, target)
        if path not in selected:
            selected.append(path)
    while len(selected) < num_frames:
        selected.append(selected[-1])
    return selected


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not groups:
        raise ValueError(f"候选文件为空: {args.candidates}")
    if args.top_k <= 0 or args.num_frames <= 0:
        raise ValueError("top-k 和 num-frames 必须大于 0")
    if not 0.0 <= args.hard_negative_iou <= args.positive_iou <= 1.0:
        raise ValueError("必须满足 0 <= hard-negative-iou <= positive-iou <= 1")

    frame_cache: dict[str, tuple[list[float], list[Path]]] = {}
    records: list[dict[str, Any]] = []
    for group in groups:
        query_id = str(group["query_id"])
        video_id = str(group["video_id"])
        if video_id not in frame_cache:
            frame_cache[video_id] = frame_index(args.frame_root, video_id)
        timestamps, paths = frame_cache[video_id]
        candidates = sorted(group.get("candidates", []), key=lambda item: int(item["rank"]))[: args.top_k]
        positives = [item for item in candidates if float(item["best_temporal_iou"]) >= args.positive_iou]
        negatives = [item for item in candidates if float(item["best_temporal_iou"]) < args.positive_iou]
        # 优先保留边界附近的窗口，再补充检索排序靠前的低 IoU 窗口。
        negatives.sort(
            key=lambda item: (
                0 if float(item["best_temporal_iou"]) >= args.hard_negative_iou else 1,
                -float(item["best_temporal_iou"]),
                int(item["rank"]),
            )
        )
        negatives = negatives[: args.max_negatives_per_query]
        selected = positives if args.include_all_positives else positives[:]
        selected += negatives
        selected.sort(key=lambda item: int(item["rank"]))
        if not selected:
            continue
        for item in selected:
            start = float(item["start"])
            end = float(item["end"])
            frame_paths = select_frame_paths(timestamps, paths, start, end, args.num_frames)
            tiou = float(item["best_temporal_iou"])
            label = int(tiou >= args.positive_iou)
            negative_type = "positive" if label else (
                "hard" if tiou >= args.hard_negative_iou else "random"
            )
            records.append(
                {
                    "id": f"{query_id}@candidate{int(item['rank']):02d}",
                    "images": [str(path.resolve()) for path in frame_paths],
                    "instruction": str(group["instruction"]),
                    "response": "1" if label else "0",
                    "label": label,
                    "task": "temporal_understanding",
                    "video_id": video_id,
                    "query_id": query_id,
                    "candidate_rank": int(item["rank"]),
                    "center": round(float(item["center"]), 3),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "retrieval_score": float(item["score"]),
                    "importance": float(item["importance"]),
                    "best_temporal_iou": tiou,
                    "gt_segments": group.get("gt_segments", []),
                    "split": group.get("split", "unknown"),
                    "negative_type": negative_type,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    records = build_records(args)
    if not records:
        raise ValueError("没有生成任何 LoRA-U 样本")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("LoRA-U 样本 id 重复")
    for record in records:
        if len(record["images"]) != args.num_frames:
            raise ValueError(f"样本 {record['id']} 帧数异常: {len(record['images'])}")
        missing = [path for path in record["images"] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"样本 {record['id']} 缺少图片: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "samples": len(records),
        "queries": len({record["query_id"] for record in records}),
        "videos": len({record["video_id"] for record in records}),
        "positives": sum(record["label"] for record in records),
        "negatives": sum(not record["label"] for record in records),
        "positive_rate": sum(record["label"] for record in records) / len(records),
        "negative_types": dict(Counter(record["negative_type"] for record in records)),
        "candidate_ranks": dict(Counter(record["candidate_rank"] for record in records)),
        "num_frames": args.num_frames,
        "top_k": args.top_k,
        "positive_iou": args.positive_iou,
        "hard_negative_iou": args.hard_negative_iou,
        "output": str(args.output),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
