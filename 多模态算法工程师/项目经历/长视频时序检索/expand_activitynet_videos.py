"""在当前 ActivityNet 子集基础上补充视频，生成 300-video 标注清单。"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-videos", type=int, default=300)
    parser.add_argument("--min-segment-seconds", type=float, default=8.0)
    parser.add_argument("--max-video-mb", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = str(record.get("video_id") or record.get("id") or "")
            if video_id:
                metadata[video_id if video_id.startswith("v_") else f"v_{video_id}"] = record
    if not metadata:
        raise ValueError(f"视频元数据为空: {path}")
    return metadata


def valid_candidates(
    annotations: Path,
    metadata: dict[str, dict[str, Any]],
    excluded: set[str],
    min_segment_seconds: float,
    max_video_mb: float,
) -> list[dict[str, Any]]:
    raw = json.loads(annotations.read_text(encoding="utf-8"))
    max_bytes = int(max_video_mb * 1024 * 1024)
    candidates: list[dict[str, Any]] = []
    for video_id, info in raw.items():
        if video_id in excluded:
            continue
        meta = metadata.get(video_id)
        if meta is None:
            continue
        size_bytes = int(meta.get("size_bytes") or 0)
        if size_bytes <= 0 or size_bytes > max_bytes:
            continue
        duration = float(info.get("duration") or 0.0)
        segments: list[dict[str, Any]] = []
        for sentence, timestamp in zip(info.get("sentences") or [], info.get("timestamps") or []):
            if len(timestamp) != 2 or not str(sentence).strip():
                continue
            start, end = map(float, timestamp)
            if duration > 0:
                start = max(0.0, min(start, duration))
                end = max(start, min(end, duration))
            if end > start and end - start >= min_segment_seconds:
                segments.append({"start": start, "end": end, "query": str(sentence).strip()})
        if not segments:
            continue
        segments.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"], item["query"]))
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
    return candidates


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
        expected = int(response.headers.get("Content-Length") or item["size_bytes"])
        video_dir.mkdir(parents=True, exist_ok=True)
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
    if args.target_videos <= 0 or args.workers <= 0:
        raise ValueError("target-videos 和 workers 必须大于 0")
    base_records = [
        json.loads(line)
        for line in args.base_subset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_ids = {str(item["video_id"]) for item in base_records}
    if len(base_ids) != len(base_records):
        raise ValueError("base-subset 中存在重复 video_id")
    if len(base_records) > args.target_videos:
        raise ValueError("base-subset 数量已经超过 target-videos")
    metadata = load_metadata(args.metadata)
    candidates = valid_candidates(
        args.annotations,
        metadata,
        base_ids,
        args.min_segment_seconds,
        args.max_video_mb,
    )
    need = args.target_videos - len(base_records)
    if len(candidates) < need:
        raise ValueError(f"可补充视频不足，需要 {need} 个，只有 {len(candidates)} 个")
    random.Random(args.seed).shuffle(candidates)
    extra = candidates[:need]
    output_dir = args.output_dir
    video_dir = output_dir / "videos"
    normalized: list[dict[str, Any]] = []
    for item in base_records:
        normalized.append({**item, "source_subset": "subset_100"})
    for item in extra:
        normalized.append(
            {
                "video_id": item["video_id"],
                "video_path": str((video_dir / f"{item['video_id']}.mp4").resolve()),
                "duration": item["duration"],
                "query": item["query"],
                "gt_segments": item["gt_segments"],
                "source": "ActivityNet Captions train",
                "source_subset": "new_video",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "activitynet_subset.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in normalized), encoding="utf-8"
    )
    (output_dir / "download_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in extra), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "target_videos": args.target_videos,
        "base_videos": len(base_records),
        "new_videos": len(extra),
        "estimated_new_bytes": sum(int(item["size_bytes"]) for item in extra),
        "estimated_new_gib": round(sum(int(item["size_bytes"]) for item in extra) / 1024**3, 3),
        "min_segment_seconds": args.min_segment_seconds,
        "max_video_mb": args.max_video_mb,
        "downloaded": False,
    }
    if args.download_videos and not args.dry_run:
        results: list[tuple[str, bool, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, item, video_dir) for item in extra]
            for future in as_completed(futures):
                results.append(future.result())
        failures = [item for item in results if not item[1]]
        summary["downloaded"] = not failures
        summary["download_success"] = len(results) - len(failures)
        summary["download_failures"] = [
            {"video_id": item[0], "error": item[2]} for item in failures
        ]
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.download_videos and not args.dry_run and summary.get("download_failures"):
        raise RuntimeError(f"有 {len(summary['download_failures'])} 个新增视频下载失败")


if __name__ == "__main__":
    main()
