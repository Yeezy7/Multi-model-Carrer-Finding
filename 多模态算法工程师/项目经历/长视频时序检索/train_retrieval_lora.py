"""Qwen3-VL Query-Frame 相关性 LoRA/QLoRA 训练入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from retrieval_data import RetrievalCollator, RetrievalDataset, load_manifest, manifest_summary


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("无法定位仓库根目录")


def latest_checkpoint(output_dir: Path) -> str | bool:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return str(sorted(checkpoints)[-1][1]) if checkpoints else False


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root()
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=repo_root / "pre-trained-model" / "Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "retrieval_lora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=512 * 32 * 32)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-batch-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("必须满足 0 < min-pixels <= max-pixels")
    train_records = load_manifest(args.train_path, require_images=not args.dry_run)
    val_records = None
    if args.val_path is not None:
        val_records = load_manifest(args.val_path, require_images=not args.dry_run)
    print(json.dumps({"train": manifest_summary(train_records), "val": manifest_summary(val_records or [])}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.model.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = args.max_length
    collator = RetrievalCollator(processor, max_length=args.max_length)
    train_dataset = RetrievalDataset(train_records)
    if args.validate_batch_only:
        batch = collator([train_dataset[0]])
        print(json.dumps({"validated_batch": {key: list(value.shape) for key, value in batch.items()}}, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA 训练需要 CUDA GPU；当前仅允许在 CPU 上做数据和 collator 校验")
    set_seed(args.seed)

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.qlora:
        model_kwargs.update(
            {
                "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if val_records else "no",
        eval_steps=args.save_steps if val_records else None,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"train_summary": manifest_summary(train_records), "val_summary": manifest_summary(val_records or [])})
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=RetrievalDataset(val_records) if val_records else None,
        data_collator=collator,
    )
    if args.resume_from_checkpoint == "auto":
        resume = latest_checkpoint(args.output_dir)
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
