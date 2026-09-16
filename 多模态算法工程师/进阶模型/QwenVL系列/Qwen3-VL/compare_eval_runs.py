"""Compare two evaluation summaries after checking protocol compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL_KEYS = (
    "min_pixels",
    "max_pixels",
    "max_new_tokens",
    "degradation_mode",
    "degradation_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    incompatible = {
        key: {"base": base.get(key), "candidate": candidate.get(key)}
        for key in PROTOCOL_KEYS
        if base.get(key) != candidate.get(key)
    }
    if incompatible:
        raise ValueError(f"评测协议不一致: {incompatible}")
    if set(base["datasets"]) != set(candidate["datasets"]):
        raise ValueError("两个 summary 的数据集集合不一致")
    comparisons = {}
    for dataset in base["datasets"]:
        base_result = base["datasets"][dataset]
        candidate_result = candidate["datasets"][dataset]
        if base_result["num_samples"] != candidate_result["num_samples"]:
            raise ValueError(f"{dataset} 样本数不一致")
        comparisons[dataset] = {
            "metric": base_result["metric"],
            "num_samples": base_result["num_samples"],
            "base_percent": base_result["score_percent"],
            f"{args.label}_percent": candidate_result["score_percent"],
            "delta_points": candidate_result["score_percent"] - base_result["score_percent"],
        }
    result = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "protocol": {key: base.get(key) for key in PROTOCOL_KEYS},
        "datasets": comparisons,
        "warning": "Only treat this as formal when both runs cover the complete fixed evaluation splits.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
