"""选择 ActivityNet Captions 小子集，并转换为本项目的视频级标注格式。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="ActivityNet Captions train.json")
    parser.add_argument("--metadata", type=Path, required=True, help="包含 video_url 的 metadata.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--min-segment-seconds", type=float, default=8.0)
    parser.add_argument("--max-video-mb", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = str(record.get("video_id", ""))
            if video_id:
                metadata[video_id if video_id.startswith("v_") else f"v_{video_id}"] = record
    if not metadata:
        raise ValueError(f"视频元数据为空: {path}")
    return metadata


def select_records(
    annotations: Path,
    metadata: dict[str, dict[str, Any]],
    num_pairs: int,
    min_segment_seconds: float,
    max_video_mb: float,
    seed: int,
) -> list[dict[str, Any]]:
    raw = json.loads(annotations.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    max_bytes = int(max_video_mb * 1024 * 1024)
    for video_id, info in raw.items():
        meta = metadata.get(video_id)
        if meta is None:
            continue
        size_bytes = int(meta.get("size_bytes") or 0)
        if size_bytes <= 0 or size_bytes > max_bytes:
            continue
        duration = float(info.get("duration") or 0.0)
        sentences = info.get("sentences") or []
        timestamps = info.get("timestamps") or []
        segments = []
        for sentence, timestamp in zip(sentences, timestamps):
            if len(timestamp) != 2:
                continue
            start, end = map(float, timestamp)
            if not str(sentence).strip() or end <= start or end - start < min_segment_seconds:
                continue
            if duration > 0:
                start = max(0.0, min(start, duration))
                end = max(start, min(end, duration))
            if end - start >= min_segment_seconds:
                segments.append({"start": start, "end": end, "query": str(sentence).strip()})
        if not segments:
            continue
        segments.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))
        selected = segments[0]
        candidates.append(
            {
                "video_id": video_id,
                "video_url": str(meta["video_url"]),
                "size_bytes": size_bytes,
                "duration": duration,
                "query": selected["query"],
                "gt_segments": [{"start": selected["start"], "end": selected["end"]}],
            }
        )
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda item: item["video_id"])
    if len(candidates) < num_pairs:
        raise ValueError(f"符合条件的视频不足 {num_pairs} 个，当前只有 {len(candidates)} 个")
    return candidates[:num_pairs]


def download_one(item: dict[str, Any], video_dir: Path) -> tuple[str, bool, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("下载视频需要 requests") from exc
    video_path = video_dir / f"{item['video_id']}.mp4"
    if video_path.is_file() and video_path.stat().st_size > 0:
        return item["video_id"], True, "exists"
    temporary = video_path.with_suffix(".mp4.part")
    try:
        response = requests.get(item["video_url"], stream=True, timeout=(20, 120))
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        expected = int(content_length) if content_length else int(item["size_bytes"])
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        actual = temporary.stat().st_size
        if actual < max(1024, min(expected, 1024 * 1024)):
            return item["video_id"], False, f"too_small:{actual}"
        temporary.replace(video_path)
        return item["video_id"], True, "downloaded"
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        return item["video_id"], False, f"{type(exc).__name__}:{exc}"


def main() -> None:
    args = parse_args()
    if args.num_pairs <= 0 or args.workers <= 0:
        raise ValueError("num-pairs 和 workers 必须大于 0")
    metadata = load_metadata(args.metadata)
    selected = select_records(
        args.annotations,
        metadata,
        args.num_pairs,
        args.min_segment_seconds,
        args.max_video_mb,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    normalized = []
    for item in selected:
        normalized.append(
            {
                "video_id": item["video_id"],
                "video_path": str((video_dir / f"{item['video_id']}.mp4").resolve()),
                "duration": item["duration"],
                "query": item["query"],
                "gt_segments": item["gt_segments"],
                "source": "ActivityNet Captions train",
            }
        )
    (args.output_dir / "activitynet_subset.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in normalized), encoding="utf-8"
    )
    (args.output_dir / "download_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected), encoding="utf-8"
    )
    summary = {
        "num_pairs": len(selected),
        "num_videos": len({item["video_id"] for item in selected}),
        "estimated_bytes": sum(item["size_bytes"] for item in selected),
        "estimated_gib": round(sum(item["size_bytes"] for item in selected) / 1024**3, 3),
        "min_segment_seconds": args.min_segment_seconds,
        "max_video_mb": args.max_video_mb,
        "downloaded": False,
    }
    if args.download_videos and not args.dry_run:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        video_dir.mkdir(parents=True, exist_ok=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, item, video_dir) for item in selected]
            for future in as_completed(futures):
                results.append(future.result())
        failures = [result for result in results if not result[1]]
        summary["downloaded"] = len(failures) == 0
        summary["download_success"] = len(results) - len(failures)
        summary["download_failures"] = [{"video_id": item[0], "error": item[2]} for item in failures]
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.download_videos and not args.dry_run and summary.get("download_failures"):
        raise RuntimeError(f"有 {len(summary['download_failures'])} 个视频下载失败，详情见 selection_summary.json")


if __name__ == "__main__":
    main()
