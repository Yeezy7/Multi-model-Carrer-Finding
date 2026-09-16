"""Single-GPU multimodal QLoRA SFT for Qwen3-VL-8B-Instruct."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

# Some inherited shell profiles accidentally set CUDA_HOME=:/usr/local/cuda.
# DeepSpeed is not used here, but Accelerate probes it during Trainer startup.
_cuda_home = os.environ.get("CUDA_HOME", "")
if ":" in _cuda_home:
    # Some shell profiles concatenate CUDA_HOME with itself using ':'.
    # DeepSpeed expects one directory and otherwise constructs an invalid nvcc path.
    _cuda_candidates = [part for part in _cuda_home.split(":") if part]
    _cuda_home = next(
        (part for part in _cuda_candidates if Path(part, "bin", "nvcc").is_file()),
        _cuda_candidates[0] if _cuda_candidates else "",
    )
    os.environ["CUDA_HOME"] = _cuda_home
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qwen3vl_triton_cache")

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from multitask_data import (
    Qwen3VLDataCollator,
    TaskSampledDataset,
    load_manifest,
    manifest_summary,
    parse_task_weights,
)


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "datasets").exists():
            return candidate
    raise RuntimeError("无法定位仓库根目录")


class DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset: TaskSampledDataset) -> None:
        """初始化数据集 epoch 回调。"""
        self.dataset = dataset
        

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """在每个训练 epoch 开始时设置数据集的 epoch，以便进行任务采样。"""
        self.dataset.set_epoch(int(state.epoch or 0))


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root()
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "pre-trained-model" / "Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "datasets" / "Qwen3VL-SFT" / "train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "checkpoints" / "qwen3vl_8b_multitask_qlora",
    )
    parser.add_argument(
        "--task-weights",
        default="docvqa=0.45,chartqa=0.30,infographicvqa=0.25",
        help="逗号分隔的 task=weight；未列出的任务权重为 0",
    )
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="0 表示清单样本总数")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="学习率预热比例")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--resolution-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=1024 * 32 * 32)
    parser.add_argument("--fixed-pixels", type=int, default=768 * 32 * 32)
    parser.add_argument("--hard-augment-probability", type=float, default=0.20, help="在训练中使用强增强的概率，范围 [0, 1]")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="启用梯度检查点以节省显存")
    parser.add_argument("--logging-steps", type=int, default=5, help="日志记录步数")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="仅打印训练配置和样本预览，不进行训练")
    parser.add_argument("--validate-batch-only", action="store_true")
    return parser.parse_args()


def resolved_pixels(args: argparse.Namespace) -> tuple[int, int]:
    if args.resolution_mode == "fixed":
        return args.fixed_pixels, args.fixed_pixels
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("动态分辨率要求 0 < min_pixels <= max_pixels")
    return args.min_pixels, args.max_pixels


def serializable_config(args: argparse.Namespace, extra: Dict[str, Any]) -> Dict[str, Any]:
    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    values.update(extra)
    return values


def latest_checkpoint(output_dir: Path) -> str | bool:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else False


def main() -> None:
    args = parse_args()
    if not 0 <= args.hard_augment_probability <= 1:
        raise ValueError("hard_augment_probability 必须位于 [0, 1]")
    records = load_manifest(args.data_path)    # 训练数据清单
    summary = manifest_summary(records)  # 
    available_tasks = sorted(summary["tasks"])
    task_weights = parse_task_weights(args.task_weights, available_tasks)
    dataset = TaskSampledDataset(
        records,    # 训练数据清单
        task_weights=task_weights,  # docvqa=0.45,chartqa=0.30,infographicvqa=0.25
        samples_per_epoch=args.samples_per_epoch,   # 默认为0，表示清单样本总数
        seed=args.seed,
    )
    print(json.dumps({"manifest": summary, "task_weights": task_weights}, ensure_ascii=False, indent=2))
    if args.dry_run:
        preview = [dataset[index]["id"] for index in range(min(10, len(dataset)))]
        print(json.dumps({"dry_run": True, "sampled_preview": preview}, ensure_ascii=False, indent=2))
        return
    if not args.model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 CUDA GPU")

    set_seed(args.seed)
    min_pixels, max_pixels = resolved_pixels(args)
    processor = AutoProcessor.from_pretrained(     # 加载模型处理器
        args.model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer.padding_side = "right"  # 设置填充方向为右侧
    processor.tokenizer.model_max_length = args.max_length
    collator = Qwen3VLDataCollator(     # 数据整理器，用于将多个样本合并为一个批次
        processor,
        max_length=args.max_length,
        hard_augment_probability=args.hard_augment_probability,
    )
    if args.validate_batch_only:    # 仅验证批次数据整理器是否正常工作
        batch = collator([dataset[0]])
        print(
            json.dumps(
                {"validated_batch": {key: list(value.shape) for key, value in batch.items()}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    model_kwargs: Dict[str, Any] = {     # 模型加载参数
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",   # 使用标准点积注意力实现
        "low_cpu_mem_usage": True,       # 在 CPU 上使用低内存加载模型
    }
    if args.qlora:  # QLoRA 量化训练
        model_kwargs.update(
            {
                # CUDA_VISIBLE_DEVICES 会重新编号可见 GPU。例如
                # CUDA_VISIBLE_DEVICES=1 之后，物理 GPU 1 在进程内编号为 0。
                # 单卡直接运行时没有 LOCAL_RANK，因此默认必须使用可见设备 0。
                "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},  # 模型加载到指定 GPU
                "quantization_config": BitsAndBytesConfig(       # 量化配置
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",      # 使用 NF4 量化类型
                    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算使用 bfloat16
                    bnb_4bit_use_double_quant=True,     # 使用双重量化
                ),
            }
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)  # 加载 Qwen3-VL 模型   默认是推理状态
    if args.qlora:
        model = prepare_model_for_kbit_training(    # 准备模型以进行 k-bit 训练
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,  # 启用梯度检查点以节省显存
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",   # 指定任务类型为因果语言建模
    )
    
    # 将 LoRA 配置应用于模型，返回一个新的模型实例，该实例具有 LoRA 层
    # 主要将 Q、K、V、O 投影层替换为 LoRA 层，以便在训练时只更新这些层的参数，从而减少训练参数量
    model = get_peft_model(model, lora_config)  
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.config.use_cache = False      # 禁用缓存以节省显存
    model.print_trainable_parameters()  # 打印可训练参数信息

    # =========== 开始训练 ==========
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # 梯度累积步数
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,  
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",   # 使用 8-bit Adam 优化器进行 QLoRA 训练，否则使用标准 AdamW 优化器
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,  # 禁用删除未使用的列，以确保数据整理器可以访问所有输入字段
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    # 保存训练配置
    config = serializable_config(
        args,
        {
            "resolved_min_pixels": min_pixels,
            "resolved_max_pixels": max_pixels,
            "manifest_summary": summary,
            "normalized_task_weights": task_weights,
            "visible_cuda_device": torch.cuda.get_device_name(0),
        },
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[DatasetEpochCallback(dataset)],  # 在每个训练 epoch 开始时设置数据集的 epoch，以便进行任务采样
    )
    if args.resume_from_checkpoint == "auto":
        resume: str | bool = latest_checkpoint(args.output_dir)
    elif args.resume_from_checkpoint.lower() in {"none", "false", "0"}:
        resume = False
    else:
        resume = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    model.config.use_cache = True
    print(f"adapter_saved: {args.output_dir}")


if __name__ == "__main__":
    main()
