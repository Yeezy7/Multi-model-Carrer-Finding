"""Reproducible zero-shot evaluation for DocVQA, ChartQA and InfographicVQA."""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from PIL import Image, ImageEnhance, ImageFilter
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from vqa_metrics import anls_score, chartqa_relaxed_accuracy


DATASET_NAMES = ("docvqa", "chartqa", "infographicvqa")
DEGRADATION_MODES = ("none", "mixed", "low_resolution", "blur", "low_contrast", "grayscale")


@dataclass
class VQASample:
    sample_id: str
    dataset: str
    split: str
    image: Image.Image
    question: str
    answers: List[str]
    category: str


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "datasets").exists():
            return candidate
    raise RuntimeError("无法从脚本位置找到仓库根目录")


def _decode_image(payload: Any) -> Image.Image:
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"不支持的图像字段类型: {type(payload).__name__}")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def _iter_parquet_rows(files: Sequence[Path], columns: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """迭代 Parquet 文件中的行，按指定的列返回字典。"""
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1, columns=columns):
            yield batch.to_pylist()[0]


def iter_docvqa(data_root: Path) -> Iterator[VQASample]:
    """迭代 DocVQA 验证集样本，返回 VQASample 对象。"""
    files = sorted((data_root / "DocVQA" / "DocVQA").glob("validation-*.parquet"))
    columns = ["questionId", "question", "image", "answers", "data_split"]
    for row in _iter_parquet_rows(files, columns):
        answers = [str(answer) for answer in (row.get("answers") or [])]
        yield VQASample(
            sample_id=str(row["questionId"]),
            dataset="docvqa",
            split="validation",
            image=_decode_image(row["image"]),
            question=str(row["question"]),
            answers=answers,
            category="all",
        )


def iter_chartqa(data_root: Path) -> Iterator[VQASample]:
    files = sorted((data_root / "ChartQA" / "data").glob("test-*.parquet"))
    columns = ["imgname", "query", "label", "type", "image"]
    for index, row in enumerate(_iter_parquet_rows(files, columns)):
        yield VQASample(
            sample_id=f"test:{index}:{row['imgname']}",
            dataset="chartqa",
            split="test",
            image=_decode_image(row["image"]),
            question=str(row["query"]),
            answers=[str(row["label"])],
            category=str(row.get("type") or "unknown"),
        )


def iter_infographicvqa(data_root: Path) -> Iterator[VQASample]:
    files = sorted((data_root / "InfographicVQA" / "data").glob("validation-*.parquet"))
    columns = ["id", "media", "messages", "answer", "question_type"]
    for row in _iter_parquet_rows(files, columns):
        messages = json.loads(row["messages"])
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages:
            raise ValueError(f"InfographicVQA 样本 {row['id']} 缺少 user question")
        media = row.get("media") or []
        if len(media) != 1:
            raise ValueError(f"InfographicVQA 样本 {row['id']} 包含 {len(media)} 张图")
        yield VQASample(
            sample_id=str(row["id"]),
            dataset="infographicvqa",
            split="validation",
            image=_decode_image(media[0]),
            question=str(user_messages[0]["question"]),
            answers=[str(answer) for answer in (row.get("answer") or [])],
            category=str(row.get("question_type") or "unknown"),
        )


def iter_samples(dataset: str, data_root: Path) -> Iterator[VQASample]:
    """根据指定的数据集名称迭代 VQASample 样本。"""
    readers = {
        "docvqa": iter_docvqa,
        "chartqa": iter_chartqa,
        "infographicvqa": iter_infographicvqa,
    }
    yield from readers[dataset](data_root)


def prompt_for(sample: VQASample) -> str:
    """为给定的 VQASample 生成提示文本，指示模型仅使用图像回答问题。"""
    visual_type = "chart" if sample.dataset == "chartqa" else "document"
    return (
        f"Answer the question using only the {visual_type} image. "
        "Return only the short answer without explanation.\n"
        f"Question: {sample.question}"
    )

def degrade_image(image: Image.Image, mode: str, seed: int) -> tuple[Image.Image, str]:
    """为评估目的对图像应用一种退化模式。"""
    rng = random.Random(seed)
    selected = rng.choice(DEGRADATION_MODES[2:]) if mode == "mixed" else mode
    if selected == "none":
        return image, selected
    if selected == "low_resolution":
        factor = 0.40
        small = image.resize(
            (max(64, round(image.width * factor)), max(64, round(image.height * factor))),
            Image.Resampling.BILINEAR,
        )
        return small.resize(image.size, Image.Resampling.BILINEAR), selected
    if selected == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0)), selected
    if selected == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(0.55), selected
    return image.convert("L").convert("RGB"), selected


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    """加载已完成的样本记录，避免重复推理。"""
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "ok":
                    completed[str(record["sample_id"])] = record
    return completed


def score_record(sample: VQASample, prediction: str) -> float:
    if sample.dataset == "chartqa":
        return chartqa_relaxed_accuracy(prediction, sample.answers[0])
    return anls_score(prediction, sample.answers)


def infer_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    sample: VQASample,
    max_new_tokens: int,
) -> str:
    """
    对单个样本进行推理，生成预测结果。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image},
                {"type": "text", "text": prompt_for(sample)},
            ],
        }
    ]
    # 使用处理器将消息转换为模型输入，添加生成提示，并返回 PyTorch 张量
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode(): # 在推理模式下禁用梯度计算，节省内存和计算资源
        # 使用模型生成预测结果，指定最大新令牌数，不进行采样，启用缓存以加速生成
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed_ids = generated_ids[:, inputs.input_ids.shape[1] :] # 去掉输入部分，只保留生成的输出
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = [record for record in records if record.get("status") == "ok"]
    scores = [float(record["score"]) for record in records]
    latencies = [float(record["latency_seconds"]) for record in records]
    by_category: Dict[str, List[float]] = {}
    for record in records:
        by_category.setdefault(str(record["category"]), []).append(float(record["score"]))
    return {
        "num_samples": len(records),
        "score": sum(scores) / len(scores) if scores else 0.0,
        "score_percent": 100 * sum(scores) / len(scores) if scores else 0.0,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_category": {
            category: {
                "num_samples": len(category_scores),
                "score_percent": 100 * sum(category_scores) / len(category_scores),
            }
            for category, category_scores in sorted(by_category.items())
        },
    }


def read_records(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件中的记录，返回一个字典列表。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_dataset(
    dataset: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    data_root: Path,
    output_dir: Path,
    max_samples: int,
    offset: int,
    max_new_tokens: int,
    fail_fast: bool,
    log_every: int,
    degradation_mode: str,
    degradation_seed: int,
) -> Dict[str, Any]:
    prediction_path = output_dir / f"{dataset}.jsonl"
    completed = load_completed(prediction_path)     # 加载已完成的样本记录，避免重复推理
    attempted = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("a", encoding="utf-8") as output:
        for source_index, sample in enumerate(iter_samples(dataset, data_root)):
            if source_index < offset:   # 如果指定了 offset，则跳过前 offset 个样本
                continue
            if max_samples and attempted >= max_samples:
                break
            attempted += 1
            if sample.sample_id in completed:
                continue

            started = time.perf_counter() # 记录推理开始时间
            try:
                # 对图像进行退化处理，并生成预测结果，默认 不启用
                degraded_image, applied_degradation = degrade_image(
                    sample.image,
                    degradation_mode,
                    degradation_seed + source_index,
                )
                evaluation_sample = replace(sample, image=degraded_image)
                prediction = infer_one(model, processor, evaluation_sample, max_new_tokens)
                record = {
                    "sample_id": sample.sample_id,
                    "source_index": source_index,
                    "dataset": dataset,
                    "split": sample.split,
                    "category": sample.category,
                    "question": sample.question,
                    "answers": sample.answers,
                    "image_width": sample.image.width,
                    "image_height": sample.image.height,
                    "degradation": applied_degradation,
                    "prediction": prediction,
                    "score": score_record(sample, prediction),
                    "latency_seconds": time.perf_counter() - started,
                    "status": "ok",
                }
            except Exception as error:
                record = {
                    "sample_id": sample.sample_id,
                    "source_index": source_index,
                    "dataset": dataset,
                    "split": sample.split,
                    "category": sample.category,
                    "question": sample.question,
                    "answers": sample.answers,
                    "image_width": sample.image.width,
                    "image_height": sample.image.height,
                    "prediction": "",
                    "score": 0.0,
                    "latency_seconds": time.perf_counter() - started,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if attempted == 1 or attempted % log_every == 0 or record["status"] == "error":
                print(
                    f"[{dataset}] {attempted}/{max_samples or 'all'} "
                    f"id={sample.sample_id} score={record['score']:.4f} "
                    f"latency={record['latency_seconds']:.2f}s status={record['status']}",
                    flush=True,
                )
            if record["status"] == "error" and fail_fast:
                raise RuntimeError(record["error"])

    summary = summarize(read_records(prediction_path))
    summary.update(
        {
            "dataset": dataset,
            "metric": "relaxed_accuracy" if dataset == "chartqa" else "anls",
            "prediction_path": str(prediction_path),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(*DATASET_NAMES, "all"),
        default=["all"],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/pre-trained-model" / "Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--data-root", type=Path, default=repo_root / "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "base_eval",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="每个数据集最多样本数，0 表示全量")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32, help="最小图像像素数，低于该值的图像将被缩放")
    parser.add_argument("--max-pixels", type=int, default=1024 * 32 * 32, help="最大图像像素数")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--adapter", type=Path, help="可选的 PEFT/QLoRA adapter 目录")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--degradation-mode", choices=DEGRADATION_MODES, default="none")
    parser.add_argument("--degradation-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(DATASET_NAMES) if "all" in args.datasets else args.datasets
    if not args.model.exists():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")

    # 定义评测协议 
    protocol = {
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "precision": "nf4" if args.load_in_4bit else "bf16",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "degradation_mode": args.degradation_mode,
        "degradation_seed": args.degradation_seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol != protocol:
            raise ValueError(
                f"输出目录已绑定其他评测协议: {protocol_path}\n"
                f"existing={existing_protocol}\nrequested={protocol}"
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # 初始化模型和处理器
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",           # 自动分配 GPU/CPU
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",  # 使用标准点积注意力，避免低精度下的数值不稳定
        "low_cpu_mem_usage": True,      # 减少 CPU 内存占用
    }
    if args.load_in_4bit:    # 如果启用 4bit 量化，则使用 BitsAndBytesConfig 配置量化参数
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    # 加载模型    
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    if args.adapter:    # 如果指定了 adapter，则加载 PEFT/QLoRA adapter
        if not args.adapter.is_dir():
            raise FileNotFoundError(f"adapter 目录不存在: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    summaries: Dict[str, Any] = {}
    for dataset in selected:
        summaries[dataset] = evaluate_dataset(
            dataset=dataset,
            model=model,
            processor=processor,
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            offset=args.offset,
            max_new_tokens=args.max_new_tokens,
            fail_fast=args.fail_fast,
            log_every=args.log_every,
            degradation_mode=args.degradation_mode,
            degradation_seed=args.degradation_seed,
        )
        print(json.dumps(summaries[dataset], ensure_ascii=False, indent=2))

    run_summary = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "precision": "nf4" if args.load_in_4bit else "bf16",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "degradation_mode": args.degradation_mode,
        "degradation_seed": args.degradation_seed,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None
        ),
        "datasets": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
