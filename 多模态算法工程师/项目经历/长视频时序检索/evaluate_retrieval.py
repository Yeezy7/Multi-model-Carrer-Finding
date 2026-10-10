"""计算 Query-Frame 检索的 Query 级和 video-macro Recall@K/AP。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser.parse_args()


def average_precision(rows: list[dict[str, Any]]) -> float:
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    positives = sum(int(row["label"]) for row in ranked)
    if positives == 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, 1):
        if int(row["label"]):
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / positives


def query_metrics(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    result = {f"recall@{k}": float(any(int(row["label"]) for row in ranked[:k])) for k in ks}
    result.update({f"precision@{k}": sum(int(row["label"]) for row in ranked[:k]) / min(k, len(ranked)) for k in ks})
    result["average_precision"] = average_precision(rows)
    return result


def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def main() -> None:
    args = parse_args()
    if any(k <= 0 for k in args.ks):
        raise ValueError("ks 必须全部大于 0")
    rows = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"打分文件为空: {args.scores}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id") or row.get("id", "").rsplit("@", 1)[0])
        grouped[query_id].append(row)
    per_query = {query_id: query_metrics(items, args.ks) for query_id, items in grouped.items()}
    by_video: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query_id, metrics in per_query.items():
        video_id = str(next(row for row in grouped[query_id]) ["video_id"])
        by_video[video_id].append(metrics)
    per_video = {video_id: mean_dict(items) for video_id, items in by_video.items()}
    summary: dict[str, Any] = {
        "scores": str(args.scores),
        "frames": len(rows),
        "queries": len(per_query),
        "videos": len(per_video),
        "positive_frames": sum(int(row["label"]) for row in rows),
        "query_macro": mean_dict(list(per_query.values())),
        "video_macro": mean_dict(list(per_video.values())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
