"""Evaluate DocVQA JSONL predictions with thresholded multi-reference ANLS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from vqa_metrics import anls_score


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: Dict[str, Any]) -> str:
    value = record.get("sample_id", record.get("id", record.get("question_id")))
    if value is None:
        raise ValueError(f"记录缺少 sample_id/id/question_id: {record}")
    return str(value)


def references(record: Dict[str, Any]) -> List[str]:
    values = record.get("answers", record.get("answer", []))
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument(
        "--gt-path",
        type=Path,
        help="可选；省略时 prediction JSONL 必须自带 answers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.pred_path)
    ground_truth = (
        {record_id(record): references(record) for record in read_jsonl(args.gt_path)}
        if args.gt_path
        else {}
    )
    scores = []
    for record in predictions:
        if record.get("status", "ok") != "ok":
            raise ValueError(f"预测包含运行失败样本: {record_id(record)}")
        expected = ground_truth.get(record_id(record), references(record))
        if not expected:
            raise ValueError(f"样本缺少参考答案: {record_id(record)}")
        prediction = str(record.get("prediction", record.get("answer", "")))
        scores.append(anls_score(prediction, expected))
    result = {
        "num_samples": len(scores),
        "anls": sum(scores) / len(scores) if scores else 0.0,
        "anls_percent": 100 * sum(scores) / len(scores) if scores else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
