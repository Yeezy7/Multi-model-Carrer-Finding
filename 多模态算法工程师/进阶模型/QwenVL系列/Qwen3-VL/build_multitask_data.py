"""Build a deduplicated Image + Instruction + Response training dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import pyarrow.parquet as pq
from PIL import Image

from multitask_data import canonical_json


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "datasets").exists():
            return candidate
    raise RuntimeError("无法定位仓库根目录")


def payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"不支持的图像字段: {type(payload).__name__}")
    return bytes(payload)


def image_extension(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
        fmt = (image.format or "png").lower()
    return {"jpeg": ".jpg", "tiff": ".tif"}.get(fmt, f".{fmt}")


def save_image(data: bytes, directory: Path, stable_id: str) -> Tuple[Path, bool]:
    digest = hashlib.sha1(data).hexdigest()[:16]
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in stable_id)
    path = directory / f"{safe_id}-{digest}{image_extension(data)}"
    if path.exists():
        return path, False
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, True


def parquet_rows(files: Iterable[Path], columns: List[str]) -> Iterator[Dict[str, Any]]:
    for path in sorted(files):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=128, columns=columns):
            yield from batch.to_pylist()


def relative_image(path: Path, output_root: Path) -> str:
    return str(path.relative_to(output_root))


def docvqa_records(data_root: Path, output_root: Path) -> Iterator[Tuple[Dict[str, Any], bool]]:
    files = (data_root / "DocVQA" / "DocVQA").glob("train-*.parquet")
    columns = ["questionId", "question", "question_types", "image", "docId", "answers"]
    for row in parquet_rows(files, columns):
        answers = [str(answer).strip() for answer in (row.get("answers") or []) if str(answer).strip()]
        if not answers:
            continue
        image_path, created = save_image(
            payload_bytes(row["image"]), output_root / "images" / "docvqa", str(row["docId"])
        )
        yield {
            "id": f"docvqa:{row['questionId']}",
            "image": relative_image(image_path, output_root),
            "instruction": str(row["question"]).strip(),
            "response": answers[0],
            "task": "docvqa",
            "source": "lmms-lab/DocVQA",
            "metadata": {"answers": answers, "question_types": row.get("question_types") or []},
        }, created


def chartqa_records(data_root: Path, output_root: Path) -> Iterator[Tuple[Dict[str, Any], bool]]:
    files = (data_root / "ChartQA" / "data").glob("train-*.parquet")
    columns = ["imgname", "query", "label", "type", "image"]
    for index, row in enumerate(parquet_rows(files, columns)):
        image_path, created = save_image(
            payload_bytes(row["image"]), output_root / "images" / "chartqa", str(row["imgname"])
        )
        yield {
            "id": f"chartqa:{index}:{row['imgname']}",
            "image": relative_image(image_path, output_root),
            "instruction": str(row["query"]).strip(),
            "response": str(row["label"]).strip(),
            "task": "chartqa",
            "source": "HuggingFaceM4/ChartQA",
            "metadata": {"question_type": str(row.get("type") or "unknown")},
        }, created


def infographic_records(data_root: Path, output_root: Path) -> Iterator[Tuple[Dict[str, Any], bool]]:
    files = list((data_root / "InfographicVQA" / "train_modelscope").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("缺少 InfographicVQA 训练 Parquet")
    image_index = 0
    for row in parquet_rows(files, ["images", "texts"]):
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError(f"InfographicVQA train row {image_index} 包含 {len(images)} 张图")
        image_path, created = save_image(
            payload_bytes(images[0]), output_root / "images" / "infographicvqa", str(image_index)
        )
        for question_index, pair in enumerate(row.get("texts") or []):
            yield {
                "id": f"infographicvqa:{image_index}:{question_index}",
                "image": relative_image(image_path, output_root),
                "instruction": str(pair["user"]).removesuffix("\nKeep it brief.").strip(),
                "response": str(pair["assistant"]).strip(),
                "task": "infographicvqa",
                "source": "AI-ModelScope/the_cauldron:infographic_vqa",
                "metadata": {},
            }, created and question_index == 0
        image_index += 1


def custom_kie_records(
    paths: Iterable[Path], output_root: Path
) -> Iterator[Tuple[Dict[str, Any], bool]]:
    for manifest_path in paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                source = json.loads(line)
                original_image = Path(source["image_path"])
                if not original_image.is_absolute():
                    original_image = manifest_path.parent / original_image
                if not original_image.is_file():
                    raise FileNotFoundError(f"{manifest_path}:{line_number}: {original_image}")
                response = source.get("response", source.get("answer"))
                response = canonical_json(response)
                image_data = original_image.read_bytes()
                image_path, created = save_image(
                    image_data,
                    output_root / "images" / "kie",
                    str(source.get("id") or f"{manifest_path.stem}-{line_number}"),
                )
                yield {
                    "id": f"kie:{source.get('id') or f'{manifest_path.stem}-{line_number}'}",
                    "image": relative_image(image_path, output_root),
                    "instruction": str(source.get("instruction", source.get("question", ""))).strip(),
                    "response": response,
                    "task": "kie",
                    "source": str(source.get("source") or manifest_path.name),
                    "metadata": source.get("metadata") or {},
                }, created


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo_root / "datasets")
    parser.add_argument("--output-root", type=Path, default=repo_root / "datasets" / "Qwen3VL-SFT")
    parser.add_argument("--tasks", nargs="+", choices=("docvqa", "chartqa", "infographicvqa"), default=list(("docvqa", "chartqa", "infographicvqa")))
    parser.add_argument("--kie-jsonl", nargs="*", type=Path, default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_root / "train.jsonl"    # 训练数据的 JSONL 格式文件
    stats_path = args.output_root / "stats.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"输出已存在: {manifest_path}；明确重建时使用 --force")
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".jsonl.incomplete") # 临时文件路径，用于在构建过程中存储中间结果，避免直接覆盖原始文件
    factories = {
        "docvqa": docvqa_records,
        "chartqa": chartqa_records,
        "infographicvqa": infographic_records,
    }
    sample_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    with temporary_path.open("w", encoding="utf-8") as output:
        for task in args.tasks:
            for record, created in factories[task](args.data_root, args.output_root):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                sample_counts[task] += 1
                image_counts[task] += int(created)
        for record, created in custom_kie_records(args.kie_jsonl, args.output_root):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_counts["kie"] += 1
            image_counts["kie"] += int(created)
    temporary_path.replace(manifest_path)
    stats = {
        "format": "Image + Instruction + Response",
        "manifest": str(manifest_path),
        "num_samples": sum(sample_counts.values()),
        "num_unique_images": sum(image_counts.values()),
        "samples_by_task": dict(sorted(sample_counts.items())),
        "images_by_task": dict(sorted(image_counts.items())),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
