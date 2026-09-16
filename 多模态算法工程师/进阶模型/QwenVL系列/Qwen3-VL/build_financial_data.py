"""把 FinQA 和 FinTabNet 转成 Qwen3-VL 的统一训练格式。

这份脚本是学习版，只保留四步：
1. 读取 FinQA；
2. 从 FinTabNet 找到对应的财报图片；
3. 读取 FinTabNet 表格结构；
4. 写出 Image + Instruction + Response JSONL。
"""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "Qwen3VL-Financial-SFT"


def clean_name(name):
    """把 FinQA 和 FinTabNet 的页名变成同一种格式。"""
    name = name.lower().replace("/", ".").replace("\\", ".")
    name = re.sub(r"\.png$", "", name)
    name = re.sub(r"_[0-9]+$", "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def save_image(image_data, output_dir, name):
    """保存图片，并返回相对输出目录的路径。"""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_hash = hashlib.sha1(image_data).hexdigest()[:16]  # 16位哈希，避免重复
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    with Image.open(io.BytesIO(image_data)) as image:
        image.verify()
        extension = ".jpg" if (image.format or "").lower() == "jpeg" else ".png"
    path = image_dir / f"{safe_name}-{image_hash}{extension}"
    if not path.exists():
        path.write_bytes(image_data)
    return str(path.relative_to(output_dir))


def read_finqa(data_dir):
    """分别读取 FinQA 的 train 和 dev。"""
    rows_by_split = {}
    for split in ("train", "dev"):
        file = data_dir / "FinQA" / f"{split}.json"
        rows_by_split[split] = []
        for item in json.loads(file.read_text(encoding="utf-8")):
            item["split"] = split
            rows_by_split[split].append(item)
    return rows_by_split


def find_finqa_images(data_dir, output_dir, finqa_rows):
    """扫描 OTSL Parquet，只保存 FinQA 需要的图片。"""
    wanted = {clean_name(row["filename"]) for row in finqa_rows}
    found = {}
    parquet_files = sorted((data_dir / "FinTabNet-OTSL").rglob("*.parquet"))

    for parquet_file in parquet_files:
        table = pq.ParquetFile(parquet_file)    # parquet读取器
        for batch in table.iter_batches(batch_size=128, columns=["filename", "image"]):
            for row in batch.to_pylist():   # 转换成Python 列表
                # [{"filename": "page_001.png", "image": {"bytes": ...}},
                # {"filename": "page_002.png", "image": {"bytes": ...}},...]
                filename = clean_name(row["filename"])
                matches = [name for name in wanted if name not in found and filename.startswith(name)]
                for name in matches:
                    found[name] = save_image(row["image"]["bytes"], output_dir, name)
            if len(found) == len(wanted):
                break
        if len(found) == len(wanted):
            break
    return found


def table_to_text(table):
    if not isinstance(table, list):
        return str(table)
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def make_finqa_records(rows, image_paths):
    records = []
    for row in rows:
        image = image_paths.get(clean_name(row["filename"]))
        answer = str(row["qa"].get("answer", "")).strip()
        if not image or not answer:
            continue

        context = " ".join(row.get("pre_text", []) + row.get("post_text", []))
        instruction = (
            "Answer the financial question using the table image and context. "
            "Return only the short answer.\n"
            f"Context: {context}\n"
            f"Table:\n{table_to_text(row.get('table_ori', row.get('table', [])))}\n"
            f"Question: {row['qa']['question']}"
        )
        records.append({
            "id": f"finqa:{row['split']}:{row['id']}",
            "image": image,
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt", "value": answer},
            ],
            "task": "finqa",
            "source": "czyssrs/FinQA",
            "metadata": {"program": row["qa"].get("program", "")},
        })
    return records


def make_fintabnet_records(data_dir, output_dir, split, max_tables=0):
    """读取 FinTabNet 的指定 split，输出表格结构 JSON。"""
    records = []
    files = sorted((data_dir / "FinTabNet-OTSL").rglob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 FinTabNet-OTSL 的 {split} parquet 文件")
    for parquet_file in files:
        table = pq.ParquetFile(parquet_file)
        columns = ["filename", "image", "otsl", "html"]
        for batch in table.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                otsl = " ".join(str(x) for x in (row.get("otsl") or []))
                html = "".join(str(x) for x in (row.get("html") or []))
                if not otsl and not html:
                    continue
                records.append({
                    "id": f"fintabnet:{split}:{row['filename']}",
                    "image": save_image(row["image"]["bytes"], output_dir, row["filename"]),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nExtract the table structure and return JSON with keys otsl and html.",
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({"otsl": otsl, "html": html}, ensure_ascii=False),
                        },
                    ],
                    "task": "kie",
                    "source": "docling-project/FinTabNet_OTSL",
                    "metadata": {"filename": row["filename"], "split": split},
                })
                if max_tables > 0 and len(records) >= max_tables:
                    return records
    return records


def write_jsonl(path, records):
    """把记录写成一行一个 JSON 的清单。"""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="构建 FinQA + FinTabNet 金融域训练数据")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-finqa", type=int, default=0, help="每个 split 最多读取多少条，0 表示全部")
    parser.add_argument("--max-tables", type=int, default=0, help="训练 split 最多读取多少条，0 表示全部")
    parser.add_argument("--max-dev-tables", type=int, default=0, help="验证 split 最多读取多少条，0 表示全部")
    parser.add_argument("--force", action="store_true", help="如果输出目录已存在，强制覆盖")
    args = parser.parse_args()

    train_manifest = args.output_dir / "train.jsonl"
    dev_manifest = args.output_dir / "dev.jsonl"
    if not args.force and (train_manifest.exists() or dev_manifest.exists()):
        raise FileExistsError(
            f"{train_manifest} 或 {dev_manifest} 已存在，请加 --force 重建"
        )

    finqa_rows = read_finqa(args.data_dir)
    if args.max_finqa > 0:
        for split in finqa_rows:
            finqa_rows[split] = finqa_rows[split][: args.max_finqa]

    all_finqa_rows = finqa_rows["train"] + finqa_rows["dev"]
    image_paths = find_finqa_images(args.data_dir, args.output_dir, all_finqa_rows)
    train_finqa_records = make_finqa_records(finqa_rows["train"], image_paths)
    dev_finqa_records = make_finqa_records(finqa_rows["dev"], image_paths)
    # 只使用官方 train/val 划分，避免把 val/test 样本放入训练集。
    train_table_records = make_fintabnet_records(
        args.data_dir, args.output_dir, "train", args.max_tables
    )
    dev_table_records = make_fintabnet_records(
        args.data_dir, args.output_dir, "val", args.max_dev_tables
    )
    train_records = train_finqa_records + train_table_records
    dev_records = dev_finqa_records + dev_table_records

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_manifest, train_records)
    write_jsonl(dev_manifest, dev_records)

    stats = {
        "train_samples": len(train_records),
        "dev_samples": len(dev_records),
        "finqa_train_samples": len(train_finqa_records),
        "finqa_dev_samples": len(dev_finqa_records),
        "fintabnet_train_samples": len(train_table_records),
        "fintabnet_dev_samples": len(dev_table_records),
        "finqa_images_found": len(image_paths),
        "finqa_images_requested": len({clean_name(row['filename']) for row in all_finqa_rows}),
        "train_manifest": str(train_manifest),
        "dev_manifest": str(dev_manifest),
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
