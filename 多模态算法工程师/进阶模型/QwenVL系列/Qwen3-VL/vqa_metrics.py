"""Benchmark metrics shared by base-model and fine-tuned-model evaluation."""

from __future__ import annotations

import string
import unicodedata
from typing import Iterable, Optional


def normalize_text(value: object) -> str:
    """Apply the light normalization used by document-VQA benchmarks."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return " ".join(text.split())


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(prediction: object, target: object) -> float:
    prediction_text = normalize_text(prediction)
    target_text = normalize_text(target)
    if not prediction_text and not target_text:
        return 1.0
    if not prediction_text or not target_text:
        return 0.0
    distance = levenshtein_distance(prediction_text, target_text)
    return 1.0 - distance / max(len(prediction_text), len(target_text))


def anls_score(
    prediction: object,
    targets: Iterable[object],
    threshold: float = 0.5,
) -> float:
    """Compute ANLS, taking the best match over valid reference answers.

    The benchmark clips similarities whose normalized edit distance is greater
    than or equal to 0.5, i.e. similarities less than or equal to 0.5.
    """
    similarities = [
        normalized_levenshtein_similarity(prediction, target)
        for target in targets
        if target is not None
    ]
    if not similarities:
        return 0.0
    best = max(similarities)
    return best if best > threshold else 0.0


def _to_float(value: object) -> tuple[Optional[float], bool]:
    text = normalize_text(value)
    is_percent = text.endswith("%")
    try:
        return float(text.removesuffix("%")), is_percent
    except ValueError:
        return None, False


def _chartqa_preprocess(value: object) -> str:
    text = normalize_text(value)
    if not any(character.isdigit() for character in text):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1]
        return text
    while text and text[-1] in string.punctuation and text[-1] != "%":
        text = text[:-1]
    return text.replace(",", "").replace("$", "")


def _numeric_match(
    prediction: float,
    prediction_is_percent: bool,
    target: float,
    target_is_percent: bool,
    tolerance: float,
) -> bool:
    def close(left: float, right: float) -> bool:
        return abs(left - right) / max(abs(right), 1e-10) <= tolerance

    if close(prediction, target):
        return True
    if prediction_is_percent or target_is_percent:
        prediction_decimal = prediction / 100 if prediction_is_percent else prediction
        target_decimal = target / 100 if target_is_percent else target
        return close(prediction_decimal, target) or close(prediction, target_decimal)
    return False


def chartqa_relaxed_accuracy(
    prediction: object,
    target: object,
    relative_tolerance: float = 0.05,
) -> float:
    """ChartQA relaxed accuracy: 5% tolerance for numbers, exact text match."""
    prediction_text = _chartqa_preprocess(prediction)
    target_text = _chartqa_preprocess(target)
    prediction_number, prediction_is_percent = _to_float(prediction_text)
    target_number, target_is_percent = _to_float(target_text)

    if prediction_number is not None and target_number is not None:
        return float(
            _numeric_match(
                prediction_number,
                prediction_is_percent,
                target_number,
                target_is_percent,
                relative_tolerance,
            )
        )
    if len(target_text) == 1 and target_text.isalpha() and prediction_text:
        return float(prediction_text[0] == target_text)
    while prediction_text and prediction_text[-1] in string.punctuation:
        prediction_text = prediction_text[:-1]
    return float(prediction_text == target_text)
