"""LoRA-R 的 Query-Frame 数据读取、拼批与提示构造。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def load_manifest(path: Path, require_images: bool = True) -> list[dict[str, Any]]:
    """读取帧级相关性 JSONL，并将图片路径解析为绝对路径。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"id", "image", "instruction", "response", "label"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {sorted(missing)}")
            image_path = Path(str(record["image"]))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if require_images and not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number} 图片不存在: {image_path}")
            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number} label 必须是 0 或 1")
            response = str(record["response"]).strip()
            if response not in {"0", "1"} or int(response) != label:
                raise ValueError(f"{path}:{line_number} response 必须与 label 一致且为 0/1")
            record["image"] = str(image_path.resolve())
            record["instruction"] = str(record["instruction"]).strip()
            record["response"] = response
            records.append(record)
    if not records:
        raise ValueError(f"训练清单为空: {path}")
    return records


class RetrievalDataset(Dataset):
    """保持清单顺序的帧级相关性数据集。"""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(record) for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def retrieval_prompt(query: str) -> str:
    return (
        "[RETRIEVAL]\n"
        "判断这张视频帧是否包含与 Query 直接相关的视觉证据。\n"
        "如果相关只输出 1，如果不相关只输出 0。不要输出其他内容。\n"
        f"Query: {query}"
    )


def _find_last_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    for start in range(len(values) - len(pattern), -1, -1):
        if list(values[start : start + len(pattern)]) == list(pattern):
            return start
    return -1


class RetrievalCollator:
    """构造 Qwen3-VL 的图像对话输入，并只对 assistant 的 0/1 计算 loss。"""

    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.assistant_prefix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_prefix_ids:
            raise RuntimeError("无法获得 Qwen3-VL assistant 模板 token")

    def _prepare_one(self, record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        with Image.open(record["image"]) as opened:
            image = opened.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": retrieval_prompt(str(record["instruction"]))},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": str(record["response"])}]},
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"样本 {record['id']} token 数 {input_ids.numel()} 超过 max_length={self.max_length}"
            )
        start = _find_last_subsequence(input_ids.tolist(), self.assistant_prefix_ids)
        if start < 0:
            raise RuntimeError(f"样本 {record['id']} 未找到 assistant response 边界")
        answer_start = start + len(self.assistant_prefix_ids)
        labels = input_ids.clone()
        labels[:answer_start] = IGNORE_INDEX
        result = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
        result["input_ids"] = input_ids
        result["labels"] = labels
        return result

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        instances = [self._prepare_one(record) for record in records]
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        batch: Dict[str, torch.Tensor] = {
            "input_ids": padded_ids,
            "labels": padded_labels,
            "attention_mask": padded_ids.ne(self.tokenizer.pad_token_id),
        }
        for key in ("pixel_values", "image_grid_thw"):
            values = [instance[key] for instance in instances if key in instance]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


def manifest_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": None,
            "videos": 0,
        }
    positives = sum(int(record["label"]) for record in records)
    return {
        "num_samples": len(records),
        "positives": positives,
        "negatives": len(records) - positives,
        "positive_rate": positives / len(records),
        "videos": len({str(record.get("video_id", record["id"])) for record in records}),
    }
