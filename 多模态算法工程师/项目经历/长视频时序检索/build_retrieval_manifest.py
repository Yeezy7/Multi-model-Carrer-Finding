"""从视频级时序标注生成 LoRA-R 的帧级 Query-Frame 清单。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="视频级标注 JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--global-fps", type=float, default=0.5)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--positive-tolerance", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-quality", type=int, default=90)
    parser.add_argument("--start-index", type=int, default=0, help="从排序后的第几个视频开始处理")
    parser.add_argument("--max-videos", type=int, default=None, help="本次最多处理多少个视频")
    parser.add_argument("--append", action="store_true", help="追加写入已有 train/val/test.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_annotations(path: Path, video_root: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"video_id", "video_path", "query", "gt_segments"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {sorted(missing)}")
            segments = []
            for segment in record["gt_segments"]:
                start = float(segment["start"])
                end = float(segment["end"])
                if start < 0 or end <= start:
                    raise ValueError(f"{path}:{line_number} 存在非法 gt segment: {segment}")
                segments.append({"start": start, "end": end})
            video_path = Path(str(record["video_path"]))
            if not video_path.is_absolute():
                video_path = (video_root or path.parent) / video_path
            records.append(
                {
                    **record,
                    "video_path": str(video_path.resolve()),
                    "query": str(record["query"]).strip(),
                    "gt_segments": segments,
                }
            )
    if not records:
        raise ValueError(f"标注为空: {path}")
    return records


def assign_splits(records: list[dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> None:
    video_ids = sorted({str(record["video_id"]) for record in records})
    if len(video_ids) < 3:
        raise ValueError(
            f"至少需要 3 个不同视频才能进行 video-disjoint train/val/test 划分，当前只有 {len(video_ids)} 个"
        )
    random.Random(seed).shuffle(video_ids)
    train_end = int(len(video_ids) * train_ratio)
    val_end = train_end + int(len(video_ids) * val_ratio)
    split_by_video = {
        video_id: "train" if index < train_end else "val" if index < val_end else "test"
        for index, video_id in enumerate(video_ids)
    }
    for record in records:
        record["split"] = split_by_video[str(record["video_id"])]


def in_gt(timestamp: float, segments: list[dict[str, float]], tolerance: float) -> bool:
    return any(segment["start"] - tolerance <= timestamp <= segment["end"] + tolerance for segment in segments)


def sample_video(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("生成帧清单需要 PyAV，请安装 av") from exc

    video_path = Path(record["video_path"])
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if args.global_fps <= 0:
        raise ValueError("global-fps 必须大于 0")
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
    except Exception as exc:
        raise RuntimeError(f"无法打开视频: {video_path}") from exc
    source_fps = float(stream.average_rate or stream.base_rate or 0.0)
    frame_count = int(stream.frames or 0)
    duration = float(record.get("duration") or 0.0)
    if frame_count <= 0 and source_fps > 0 and duration > 0:
        # 部分公开 MP4 的 stts 元数据不提供 nb_frames，但仍可顺序解码。
        # 用标注时长估算索引范围，避免把可解码视频误判为损坏。
        frame_count = max(1, int(round(duration * source_fps)))
    if source_fps <= 0 or frame_count <= 0:
        container.close()
        raise RuntimeError(f"无法读取视频元数据: {video_path}")
    if duration <= 0:
        duration = frame_count / source_fps
    step = 1.0 / args.global_fps
    timestamps = []
    current = 0.0
    while current < duration:
        timestamps.append(current)
        current += step
    target_indices = {min(frame_count - 1, max(0, round(timestamp * source_fps))): timestamp for timestamp in timestamps}

    frame_root = args.output_dir / "frames" / str(record["video_id"])
    samples: list[dict[str, Any]] = []
    frame_index = 0
    try:
        decoded_frames = container.decode(stream)
        for frame_index, frame in enumerate(decoded_frames):
            timestamp = target_indices.get(frame_index)
            if timestamp is None:
                continue
            positive = in_gt(timestamp, record["gt_segments"], args.positive_tolerance)
            image_path = frame_root / f"{timestamp:012.3f}.jpg"
            if not args.dry_run:
                frame_root.mkdir(parents=True, exist_ok=True)
                frame.to_image().save(image_path, format="JPEG", quality=args.image_quality)
            samples.append(
                {
                    "id": f"{record['video_id']}@{timestamp:.3f}",
                    "image": str(image_path.resolve()),
                    "instruction": record["query"],
                    "response": "1" if positive else "0",
                    "label": int(positive),
                    "task": "retrieval",
                    "video_id": record["video_id"],
                    "timestamp": round(timestamp, 3),
                    "duration": duration,
                    "gt_segments": record["gt_segments"],
                    "split": record["split"],
                }
            )
    finally:
        container.close()
    return samples


def downsample_negatives(samples: list[dict[str, Any]], ratio: float, seed: int) -> list[dict[str, Any]]:
    if ratio < 0:
        raise ValueError("negative-ratio 不能小于 0")
    positives = [sample for sample in samples if sample["label"] == 1]
    negatives = [sample for sample in samples if sample["label"] == 0]
    if not positives:
        raise ValueError(f"样本 {samples[0]['video_id']} 没有正帧，无法构造相关性训练样本")
    limit = min(len(negatives), max(1, round(len(positives) * ratio)))
    random.Random(seed).shuffle(negatives)
    return positives + negatives[:limit]


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
        raise ValueError("train-ratio/val-ratio 范围非法")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio 必须小于 1")
    records = load_annotations(args.annotations, args.video_root)
    assign_splits(records, args.seed, args.train_ratio, args.val_ratio)
    if args.start_index < 0 or args.start_index >= len(records):
        raise ValueError(f"start-index 超出范围: {args.start_index}, records={len(records)}")
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("max-videos 必须大于 0")
    end_index = len(records) if args.max_videos is None else min(len(records), args.start_index + args.max_videos)
    records = records[args.start_index:end_index]
    all_samples: list[dict[str, Any]] = []
    for record in records:
        samples = sample_video(record, args)
        all_samples.extend(downsample_negatives(samples, args.negative_ratio, args.seed))
    summary = {
        "annotations": len(records),
        "samples": len(all_samples),
        "videos": len({sample["video_id"] for sample in all_samples}),
        "splits": {split: sum(sample["split"] == split for sample in all_samples) for split in ("train", "val", "test")},
        "global_fps": args.global_fps,
        "negative_ratio": args.negative_ratio,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {split: [] for split in ("train", "val", "test")}
    for sample in all_samples:
        grouped[sample["split"]].append(sample)
    for split, split_samples in grouped.items():
        mode = "a" if args.append else "w"
        with (output_dir / f"{split}.jsonl").open(mode, encoding="utf-8") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
