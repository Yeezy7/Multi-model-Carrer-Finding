"""从 Query-Frame 分数生成候选时间窗口并评估 Temporal IoU。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="score_retrieval.py 输出的 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="每个 Query 的候选窗口 JSONL")
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--window-seconds", type=float, default=16.0)
    parser.add_argument("--nms-seconds", type=float, default=4.0)
    parser.add_argument("--smooth-radius", type=float, default=2.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.25)
    return parser.parse_args()


def temporal_iou(window: tuple[float, float], segment: tuple[float, float]) -> float:
    start = max(window[0], segment[0])
    end = min(window[1], segment[1])
    intersection = max(0.0, end - start)
    union = max(window[1], segment[1]) - min(window[0], segment[0])
    return intersection / union if union > 0 else 0.0


def best_iou(window: tuple[float, float], segments: list[tuple[float, float]]) -> float:
    return max((temporal_iou(window, segment) for segment in segments), default=0.0)


def build_candidates(
    rows: list[dict[str, Any]],
    window_seconds: float,
    nms_seconds: float,
    smooth_radius: float,
    smooth_alpha: float,
    max_k: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Query 没有帧分数")
    duration = max(float(row.get("duration") or 0.0) for row in rows)
    duration = max(duration, max(float(row["timestamp"]) for row in rows))
    segments = [
        (float(segment["start"]), float(segment["end"]))
        for segment in rows[0].get("gt_segments", [])
    ]
    if not segments:
        raise ValueError(f"Query {rows[0].get('query_id')} 没有 gt_segments")

    timestamps = [float(row["timestamp"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    importance: list[float] = []
    for timestamp, score in zip(timestamps, scores):
        neighbors = [
            neighbor_score
            for neighbor_timestamp, neighbor_score in zip(timestamps, scores)
            if abs(neighbor_timestamp - timestamp) <= smooth_radius
        ]
        local_mean = sum(neighbors) / len(neighbors)
        importance.append((1.0 - smooth_alpha) * score + smooth_alpha * local_mean)

    order = sorted(
        range(len(rows)),
        key=lambda index: (importance[index], scores[index], -timestamps[index]),
        reverse=True,
    )
    selected: list[int] = []
    for index in order:
        if all(abs(timestamps[index] - timestamps[other]) >= nms_seconds for other in selected):
            selected.append(index)
        if len(selected) >= max_k:
            break

    half_window = window_seconds / 2.0
    candidates: list[dict[str, Any]] = []
    for rank, index in enumerate(selected, 1):
        center = timestamps[index]
        start = max(0.0, center - half_window)
        end = min(duration, center + half_window)
        window = (start, end)
        candidates.append(
            {
                "rank": rank,
                "center": round(center, 3),
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(scores[index], 8),
                "importance": round(importance[index], 8),
                "best_temporal_iou": round(best_iou(window, segments), 8),
            }
        )
    return candidates


def aggregate_metrics(
    query_results: list[dict[str, Any]], top_ks: list[int]
) -> dict[str, Any]:
    query_metrics: dict[str, dict[str, float]] = {}
    for result in query_results:
        candidates = result["candidates"]
        metrics: dict[str, float] = {}
        for k in top_ks:
            selected = candidates[:k]
            best = max((float(item["best_temporal_iou"]) for item in selected), default=0.0)
            metrics[f"mean_best_tiou@{k}"] = best
            metrics[f"hit@{k}"] = float(best > 0.0)
            for threshold in (0.1, 0.3, 0.5):
                metrics[f"recall@{k}_tiou{threshold}"] = float(best >= threshold)
        query_metrics[str(result["query_id"])] = metrics

    def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {}
        keys = items[0].keys()
        return {key: sum(item[key] for item in items) / len(items) for key in keys}

    by_video: dict[str, list[dict[str, float]]] = defaultdict(list)
    for result in query_results:
        by_video[str(result["video_id"])].append(query_metrics[str(result["query_id"])])
    return {
        "queries": len(query_results),
        "videos": len(by_video),
        "query_macro": mean_dict(list(query_metrics.values())),
        "video_macro": mean_dict([mean_dict(items) for items in by_video.values()]),
    }


def main() -> None:
    args = parse_args()
    if any(k <= 0 for k in args.top_k):
        raise ValueError("top-k 必须全部大于 0")
    if args.window_seconds <= 0 or args.nms_seconds < 0:
        raise ValueError("window-seconds 必须大于 0，nms-seconds 不能小于 0")
    if not 0.0 <= args.smooth_alpha <= 1.0 or args.smooth_radius < 0:
        raise ValueError("smooth-alpha 应在 [0, 1] 内，smooth-radius 不能小于 0")
    rows = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"分数文件为空: {args.scores}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id") or row.get("id", "").rsplit("@", 1)[0])
        grouped[query_id].append(row)
    top_ks = sorted(set(args.top_k))
    max_k = max(top_ks)
    query_results: list[dict[str, Any]] = []
    for query_id, query_rows in grouped.items():
        query_rows.sort(key=lambda row: float(row["timestamp"]))
        candidates = build_candidates(
            query_rows,
            args.window_seconds,
            args.nms_seconds,
            args.smooth_radius,
            args.smooth_alpha,
            max_k,
        )
        query_results.append(
            {
                "query_id": query_id,
                "video_id": str(query_rows[0]["video_id"]),
                "instruction": query_rows[0].get("instruction"),
                "gt_segments": query_rows[0].get("gt_segments", []),
                "candidates": candidates,
            }
        )

    query_results.sort(key=lambda result: (result["video_id"], result["query_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in query_results),
        encoding="utf-8",
    )
    metrics = aggregate_metrics(query_results, top_ks)
    metrics.update(
        {
            "scores": str(args.scores),
            "output": str(args.output),
            "top_k": top_ks,
            "window_seconds": args.window_seconds,
            "nms_seconds": args.nms_seconds,
            "smooth_radius": args.smooth_radius,
            "smooth_alpha": args.smooth_alpha,
        }
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
