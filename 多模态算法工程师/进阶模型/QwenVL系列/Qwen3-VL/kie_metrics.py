"""Metrics for JSON structured key information extraction."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Tuple


def normalize_value(value: Any) -> str:
    text = str(value).casefold().strip()
    return re.sub(r"\s+", " ", text)


def flatten_json(value: Any, prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_json(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flattened.update(flatten_json(child, f"{prefix}[{index}]"))
    else:
        flattened[prefix] = normalize_value(value)
    return flattened


def parse_json_output(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def field_counts(prediction: Any, reference: Any) -> Tuple[int, int, int]:
    predicted = flatten_json(prediction)
    expected = flatten_json(reference)
    true_positive = sum(
        1 for key, value in predicted.items() if key in expected and expected[key] == value
    )
    false_positive = len(predicted) - true_positive
    false_negative = len(expected) - true_positive
    return true_positive, false_positive, false_negative


def micro_f1(counts: Iterable[Tuple[int, int, int]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for current_tp, current_fp, current_fn in counts:
        tp += current_tp
        fp += current_fp
        fn += current_fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_percent": 100 * f1,
    }
