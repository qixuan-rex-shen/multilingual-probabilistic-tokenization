"""Task metrics used identically by the BPE and fused downstream arms."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(predictions: Iterable[int], labels: Iterable[int]) -> dict[str, float]:
    """Return accuracy and macro F1 for sentence-level classification."""

    predicted = list(predictions)
    expected = list(labels)
    if not expected:
        raise ValueError("Cannot score an empty classification split.")
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(expected, predicted, average="micro", zero_division=0)),
    }


def token_classification_metrics(
    predictions: Iterable[Iterable[int]], labels: Iterable[Iterable[int]]
) -> dict[str, float]:
    """Return token accuracy and macro F1 while excluding ignored positions."""

    flattened_predictions: list[int] = []
    flattened_labels: list[int] = []
    for predicted_row, expected_row in zip(predictions, labels):
        for predicted, expected in zip(predicted_row, expected_row):
            if expected == -100:
                continue
            flattened_predictions.append(int(predicted))
            flattened_labels.append(int(expected))
    if not flattened_labels:
        raise ValueError("No labeled token positions were available for metric computation.")
    return {
        "accuracy": float(accuracy_score(flattened_labels, flattened_predictions)),
        "macro_f1": float(f1_score(flattened_labels, flattened_predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(flattened_labels, flattened_predictions, average="micro", zero_division=0)),
        "labeled_tokens": float(len(flattened_labels)),
    }
