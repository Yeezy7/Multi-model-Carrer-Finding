"""在已选定的视频集合上保留多个 ActivityNet Captions Query。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="ActivityNet Captions train.json")
    parser.add_argument("--base-subset", type=Path, required=True, help="当前 100-video activitynet_subset.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-video", type=int, default=3)
    parser.add_argument("--min-segment-seconds", type=float, default=8.0)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    args = parse_args()
    if args.queries_per_video <= 0:
        raise ValueError("queries-per-video 必须大于 0")
    base_records = [
        json.loads(line)
        for line in args.base_subset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not base_records:
        raise ValueError(f"基础子集为空: {args.base_subset}")
    raw = json.loads(args.annotations.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    per_video: dict[str, int] = {}
    for base in base_records:
        video_id = str(base["video_id"])
        info = raw.get(video_id)
        if info is None:
            raise KeyError(f"原始标注中找不到视频: {video_id}")
        duration = float(info.get("duration") or base.get("duration") or 0.0)
        candidates: list[dict[str, Any]] = []
        seen_queries: set[str] = set()
        for sentence, timestamp in zip(info.get("sentences") or [], info.get("timestamps") or []):
            if len(timestamp) != 2 or not str(sentence).strip():
                continue
            start, end = map(float, timestamp)
            if duration > 0:
                start = max(0.0, min(start, duration))
                end = max(start, min(end, duration))
            if end <= start or end - start < args.min_segment_seconds:
                continue
            query = str(sentence).strip()
            normalized = normalize_text(query)
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            candidates.append({"query": query, "start": start, "end": end})
        candidates.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"], item["query"]))
        candidates = candidates[: args.queries_per_video]
        if not candidates:
            raise ValueError(f"视频没有满足时长要求的 Query: {video_id}")
        per_video[video_id] = len(candidates)
        for query_index, item in enumerate(candidates):
            selected.append(
                {
                    "pair_id": f"{video_id}#q{query_index:02d}",
                    "query_id": f"{video_id}#q{query_index:02d}",
                    "video_id": video_id,
                    "video_path": base["video_path"],
                    "duration": duration,
                    "query": item["query"],
                    "gt_segments": [{"start": item["start"], "end": item["end"]}],
                    "source": "ActivityNet Captions train",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    summary = {
        "num_videos": len(base_records),
        "num_pairs": len(selected),
        "queries_per_video_requested": args.queries_per_video,
        "queries_per_video_actual_min": min(per_video.values()),
        "queries_per_video_actual_max": max(per_video.values()),
        "min_segment_seconds": args.min_segment_seconds,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
