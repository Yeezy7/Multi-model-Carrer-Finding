"""Evaluate JSON structured extraction predictions with field-level micro F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kie_metrics import field_counts, micro_f1, parse_json_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-key", default="prediction")
    parser.add_argument("--reference-key", default="response")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = []
    total = valid_json = 0
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            reference_raw = record[args.reference_key]
            reference = (
                json.loads(reference_raw) if isinstance(reference_raw, str) else reference_raw
            )
            try:
                prediction = parse_json_output(str(record[args.prediction_key]))
            except (json.JSONDecodeError, TypeError):
                counts.append(field_counts({}, reference))
                continue
            valid_json += 1
            counts.append(field_counts(prediction, reference))
    result = micro_f1(counts)
    result.update(
        {
            "num_samples": total,
            "valid_json_samples": valid_json,
            "json_valid_rate": valid_json / total if total else 0.0,
        }
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
