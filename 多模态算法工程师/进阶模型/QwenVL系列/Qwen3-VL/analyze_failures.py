"""Create an auditable failure summary from VQA prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
RELATION_WORDS = re.compile(
    r"\b(sum|difference|more|less|higher|lower|largest|smallest|between|total|average|ratio)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_type(record: Dict[str, Any]) -> str:
    question = str(record.get("question", ""))
    answers = " ".join(str(value) for value in record.get("answers", []))
    if RELATION_WORDS.search(question):
        return "cross_region_or_numeric_reasoning_candidate"
    if NUMBER_PATTERN.fullmatch(answers.strip()):
        return "numeric_reading_candidate"
    if record.get("dataset") in {"docvqa", "infographicvqa"}:
        return "ocr_or_layout_candidate"
    return "answer_mismatch"


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    successful = [record for record in records if record.get("status") == "ok"]
    failures = [record for record in successful if float(record.get("score", 0)) < 1.0]
    runtime_errors = [record for record in records if record.get("status") != "ok"]
    by_category: Dict[str, List[float]] = defaultdict(list)
    for record in successful:
        by_category[str(record.get("category", "unknown"))].append(float(record["score"]))
    candidates = Counter(candidate_type(record) for record in failures)
    return {
        "num_records": len(records),
        "num_runtime_errors": len(runtime_errors),
        "num_nonperfect_answers": len(failures),
        "nonperfect_rate": len(failures) / len(successful) if successful else 0.0,
        "failure_candidates": dict(sorted(candidates.items())),
        "by_category": {
            category: {
                "num_samples": len(scores),
                "mean_score_percent": 100 * sum(scores) / len(scores),
            }
            for category, scores in sorted(by_category.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-dataset", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    selected_cases: List[Dict[str, Any]] = []
    for path in sorted(args.prediction_dir.glob("*.jsonl")):
        records = read_jsonl(path)
        dataset = path.stem
        report[dataset] = summarize(records)
        failures = [
            record
            for record in records
            if record.get("status") == "ok" and float(record.get("score", 0)) < 1.0
        ]
        failures.sort(key=lambda record: (float(record.get("score", 0)), str(record.get("sample_id"))))
        for record in failures[: args.examples_per_dataset]:
            selected = dict(record)
            selected["candidate_type"] = candidate_type(record)
            selected["diagnostic_note"] = "启发式候选标签，需要结合原图人工复核，不能视为已证明根因。"
            selected_cases.append(selected)
    (args.output_dir / "failure_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as output:
        for record in selected_cases:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
