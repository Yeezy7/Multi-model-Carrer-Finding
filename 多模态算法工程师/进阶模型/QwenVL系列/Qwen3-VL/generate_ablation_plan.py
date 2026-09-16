"""Generate a one-factor-at-a-time training ablation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def experiment(name: str, **overrides: Any) -> Dict[str, Any]:
    base = {
        "resolution_mode": "dynamic",
        "min_pixels": 256 * 32 * 32,
        "max_pixels": 1024 * 32 * 32,
        "lora_r": 16,
        "lora_alpha": 32,
        "task_weights": "docvqa=0.45,chartqa=0.30,infographicvqa=0.25",
        "hard_augment_probability": 0.20,
        "seed": 42,
    }
    base.update(overrides)
    return {"name": name, "config": base}


def command_for(item: Dict[str, Any], output_root: Path) -> str:
    config = item["config"]
    parts = [
        "CUDA_VISIBLE_DEVICES=1 .venv/bin/python train_sft.py",
        f"--output-dir {output_root / item['name']}",
    ]
    for key, value in config.items():
        parts.append(f"--{key.replace('_', '-')} {value}")
    return " \\\n+  ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/ablations/plan.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output.parent / "checkpoints"
    natural = "docvqa=39463,chartqa=28299,infographicvqa=10074"
    plan: List[Dict[str, Any]] = [
        experiment("main_dynamic_r16_balanced"),
        experiment(
            "resolution_fixed_512",
            resolution_mode="fixed",
            fixed_pixels=512 * 32 * 32,
        ),
        experiment(
            "resolution_fixed_768",
            resolution_mode="fixed",
            fixed_pixels=768 * 32 * 32,
        ),
        experiment("lora_r8", lora_r=8, lora_alpha=16),
        experiment("lora_r32", lora_r=32, lora_alpha=64),
        experiment("sampling_docvqa_only", task_weights="docvqa=1"),
        experiment("sampling_natural_mix", task_weights=natural),
        experiment("augmentation_off", hard_augment_probability=0.0),
    ]
    document = {
        "design": "one-factor-at-a-time",
        "control": "main_dynamic_r16_balanced",
        "experiments": [
            {**item, "command": command_for(item, output_root)} for item in plan
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(plan)} experiments)")


if __name__ == "__main__":
    main()
