"""Dataset, sampling and collation utilities for Qwen3-VL document SFT."""

from __future__ import annotations

import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def canonical_json(value: Any) -> str:
    """以确定性方式序列化结构化目标。"""
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, (dict, list)):
        raise ValueError("结构化抽取 response 必须是 JSON object 或 array")
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """读取 Qwen3-VL 官方 JSONL 格式的训练清单，并进行基本验证。"""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):  #  
            if not line.strip():    # 跳过空行
                continue
            record = json.loads(line)
            # Qwen3-VL 官方格式：image + conversations。
            # 训练内部仍使用 instruction/response，便于任务采样和 loss mask。
            if "conversations" in record:
                conversations = record["conversations"]
                human = next(
                    (turn for turn in conversations if turn.get("from") in {"human", "user"}),
                    None,
                )
                assistant = next(
                    (turn for turn in conversations if turn.get("from") in {"gpt", "assistant"}),
                    None,
                )
                if human is None or assistant is None:
                    raise ValueError(f"{path}:{line_number} conversations 缺少 human/gpt 对话")
                user_text = str(human.get("value", ""))
                if "<image>" in user_text:
                    user_text = user_text.replace("<image>", "", 1)
                record["instruction"] = user_text.strip()
                record["response"] = str(assistant.get("value", "")).strip()
                record.setdefault("task", "vqa")
                record.setdefault("id", f"{path.stem}:{line_number}")
            missing = {"id", "image", "instruction", "response", "task"} - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {sorted(missing)}")
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number} 图像不存在: {image_path}")
            record["image"] = str(image_path.resolve())
            record["instruction"] = str(record["instruction"]).strip()
            record["response"] = str(record["response"]).strip()
            record["task"] = str(record["task"]).lower().strip()
            if not record["instruction"] or not record["response"]:
                raise ValueError(f"{path}:{line_number} instruction/response 不能为空")
            if record["task"] == "kie":
                record["response"] = canonical_json(record["response"])
            records.append(record)
    if not records:
        raise ValueError(f"训练清单为空: {path}")
    return records


def parse_task_weights(value: str, available_tasks: Sequence[str]) -> Dict[str, float]:
    """解析任务采样权重字符串，返回归一化的任务权重字典。"""
    available = set(available_tasks)
    if not value:
        return {task: 1.0 / len(available) for task in sorted(available)}
    weights: Dict[str, float] = {task: 0.0 for task in available}
    for part in value.split(","):
        name, separator, raw_weight = part.partition("=")
        if not separator:
            raise ValueError(f"无效 task weight: {part!r}，应为 task=weight")
        task = name.strip().lower()
        weight = float(raw_weight)
        if task not in available:
            raise ValueError(f"task weight 指向不存在的任务 {task!r}，可用任务: {sorted(available)}")
        if weight < 0:
            raise ValueError(f"task weight 不能小于 0: {part!r}")
        weights[task] = weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("至少一个 task weight 必须大于 0")
    return {task: weight / total for task, weight in weights.items() if weight > 0}


class TaskSampledDataset(Dataset):
    """按任务加权抽样的数据集。"""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],  # 训练数据清单
        task_weights: Mapping[str, float],  # docvqa=0.45,chartqa=0.30,infographicvqa=0.25
        samples_per_epoch: int,             # 默认为0，表示清单样本总数
        seed: int,
    ) -> None:
        self.by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            self.by_task[record["task"]].append(record)
        self.tasks = sorted(task_weights)
        self.weights = [float(task_weights[task]) for task in self.tasks]
        self.samples_per_epoch = samples_per_epoch or len(records)
        self.seed = seed
        self.epoch = 0
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch 必须大于 0")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        task = rng.choices(self.tasks, weights=self.weights, k=1)[0]
        record = dict(rng.choice(self.by_task[task]))
        record["sample_seed"] = rng.randrange(2**31)
        return record


def task_prompt(record: Mapping[str, Any]) -> str:
    """根据记录中的任务信息生成提示。"""
    task = str(record["task"])
    instruction = str(record["instruction"])
    if task == "chartqa":   # 图表问答
        # 仅依据图表回答问题。请仅提供简短的答案。
        prefix = "Answer the question using only the chart. Return only the short answer."
    elif task == "kie":  # 结构化信息提取
        # 仅从文档中提取请求的字段。请仅返回一个有效的 JSON 值，包含恰好请求的键，并且不包含 Markdown。
        prefix = (
            "Extract the requested fields from the document. Return only one valid JSON value "
            "with exactly the requested keys and no Markdown."
        )
    else:   # docvqa、infographicvqa、vqa 等文档问答任务
        # 仅依据文档回答问题。请仅提供简短的答案。
        prefix = "Answer the question using only the document image. Return only the short answer."
    return f"{prefix}\nQuestion: {instruction}"


def augment_hard_document(image: Image.Image, seed: int, probability: float) -> Image.Image:
    """对训练图像应用一次确定性的扫描/小文本降质处理。"""
    if probability <= 0:    # 如果概率为0或负数，则不进行增强
        return image
    rng = random.Random(seed)
    if rng.random() >= probability:    # 如果随机数大于等于概率，则不进行增强
        return image
    mode = rng.choice(("low_resolution", "blur", "low_contrast", "grayscale"))  # 随机选择一种增强模式 [低分辨率、模糊、低对比度、灰度]
    if mode == "low_resolution":   
        # 如果选择低分辨率模式，则将图像缩小到原始尺寸的30%到55%之间，然后再放大回原始尺寸，从而模拟低分辨率效果
        factor = rng.uniform(0.30, 0.55)
        width = max(64, round(image.width * factor))
        height = max(64, round(image.height * factor))
        return image.resize((width, height), Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BILINEAR
        )
    if mode == "blur":
        # 如果选择模糊模式，则对图像应用高斯模糊，模糊半径在0.6到1.3之间随机选择
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.3)))
    if mode == "low_contrast":
        """如果选择低对比度模式，则使用ImageEnhance.Contrast增强图像的对比度，增强因子在0.45到0.70之间随机选择，从而降低图像的对比度"""
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.45, 0.70))
    return image.convert("L").convert("RGB")


def _find_last_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    """在 values 中查找 pattern 的最后一次出现，返回起始索引，如果未找到则返回 -1。"""
    for start in range(len(values) - len(pattern), -1, -1):
        if list(values[start : start + len(pattern)]) == list(pattern):
            return start
    return -1


class Qwen3VLDataCollator:
    """创建多模态批次，并仅基于助手的回答来训练损失值。"""

    def __init__(
        self,
        processor: Any,
        max_length: int,
        hard_augment_probability: float,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.hard_augment_probability = hard_augment_probability
        self.assistant_prefix_ids = self.tokenizer.encode(      # 获取助手前缀的 token ID   
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_prefix_ids:
            raise RuntimeError("无法获得 assistant 模板 token")

    def _prepare_one(self, record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """准备单样本的输入和标签。
           返回：{文本提示+占位符，图像像素}
        """
        with Image.open(record["image"]) as opened:
            image = opened.convert("RGB")
        image = augment_hard_document(
            image,
            int(record.get("sample_seed", 0)),
            self.hard_augment_probability,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task_prompt(record)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(record["response"])}],
            },
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,    # 不添加生成提示
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"样本 {record['id']} token 数 {input_ids.numel()} 超过 max_length={self.max_length}; "
                "请提高 max_length 或降低 max_pixels"
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
        """将多个样本合并为一个批次。"""
        instances = [self._prepare_one(record) for record in records]
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        padded_ids = torch.nn.utils.rnn.pad_sequence(   # 将输入 ID 序列填充为相同长度的张量，使用 tokenizer 的 pad_token_id 进行填充
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
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


def manifest_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(record["task"]) for record in records)
    return {"num_samples": len(records), "tasks": dict(sorted(counts.items()))}
