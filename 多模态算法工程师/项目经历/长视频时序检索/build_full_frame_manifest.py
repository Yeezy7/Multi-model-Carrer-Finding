"""利用已生成的 0.5 FPS JPG 帧，构建不做负样本下采样的评测清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="多 Query 视频级 JSONL")
    parser.add_argument("--split-manifest", type=Path, required=True, help="已有 test.jsonl，用于确定 video-disjoint test 视频")
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-tolerance", type=float, default=0.5)
    parser.add_argument("--hard-negative-window", type=float, default=4.0)
    return parser.parse_args()


def in_segments(timestamp: float, segments: list[dict[str, Any]], tolerance: float) -> bool:
    return any(
        float(segment["start"]) - tolerance <= timestamp <= float(segment["end"]) + tolerance
        for segment in segments
    )


def main() -> None:
    args = parse_args()
    if args.positive_tolerance < 0 or args.hard_negative_window < 0:
        raise ValueError("tolerance/window 不能小于 0")
    split_rows = [
        json.loads(line)
        for line in args.split_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_video_ids = {str(row["video_id"]) for row in split_rows}
    if not test_video_ids:
        raise ValueError("split manifest 为空")

    annotations = [
        json.loads(line)
        for line in args.annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [row for row in annotations if str(row["video_id"]) in test_video_ids]
    query_ids = {str(row.get("query_id") or row.get("pair_id")) for row in queries}
    if len(query_ids) != len(queries):
        raise ValueError("test Query 存在重复 query_id")

    records: list[dict[str, Any]] = []
    missing_videos: list[str] = []
    for query in queries:
        video_id = str(query["video_id"])
        frame_dir = args.frame_root / video_id
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            missing_videos.append(video_id)
            continue
        query_id = str(query.get("query_id") or query.get("pair_id"))
        duration = float(query.get("duration") or 0.0)
        boundaries = [
            float(boundary)
            for segment in query["gt_segments"]
            for boundary in (segment["start"], segment["end"])
        ]
        for frame_path in frame_paths:
            timestamp = float(frame_path.stem)
            positive = in_segments(timestamp, query["gt_segments"], args.positive_tolerance)
            hard_negative = (
                not positive
                and any(abs(timestamp - boundary) <= args.hard_negative_window for boundary in boundaries)
            )
            records.append(
                {
                    "id": f"{query_id}@{timestamp:.3f}",
                    "image": str(frame_path.resolve()),
                    "instruction": str(query["query"]),
                    "response": "1" if positive else "0",
                    "label": int(positive),
                    "task": "retrieval",
                    "video_id": video_id,
                    "query_id": query_id,
                    "timestamp": round(timestamp, 3),
                    "duration": duration,
                    "gt_segments": query["gt_segments"],
                    "split": "test",
                    "negative_type": "positive" if positive else "hard" if hard_negative else "random",
                }
            )

    if missing_videos:
        raise FileNotFoundError(f"缺少帧目录: {missing_videos[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "queries": len(query_ids),
        "videos": len(test_video_ids),
        "samples": len(records),
        "positive_samples": sum(record["label"] for record in records),
        "random_negative_samples": sum(record["negative_type"] == "random" for record in records),
        "hard_negative_samples": sum(record["negative_type"] == "hard" for record in records),
        "duplicate_ids": len(records) - len({record["id"] for record in records}),
        "missing_images": sum(not Path(record["image"]).is_file() for record in records),
        "global_fps": 0.5,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
