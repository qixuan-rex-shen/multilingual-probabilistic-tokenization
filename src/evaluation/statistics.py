"""Reproducible seed-level and paired-prediction statistical comparisons."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
from scipy import stats


def summarize_seed_scores(scores: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    """Report mean, sample deviation, and a t-based confidence interval."""

    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot summarize zero seed scores.")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size <= 1:
        lower = upper = mean
    else:
        margin = float(stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1) * standard_deviation / math.sqrt(values.size))
        lower, upper = mean - margin, mean + margin
    return {
        "count": float(values.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence_level": float(confidence),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
    }


def paired_seed_test(baseline_scores: Iterable[float], proposed_scores: Iterable[float]) -> dict[str, float]:
    """Paired t-test for identically seeded experimental runs."""

    baseline = np.asarray(list(baseline_scores), dtype=float)
    proposed = np.asarray(list(proposed_scores), dtype=float)
    if baseline.shape != proposed.shape or baseline.size == 0:
        raise ValueError("Paired seed tests require equal non-empty score vectors.")
    if baseline.size == 1:
        return {"paired_t_statistic": float("nan"), "paired_t_p_value": float("nan"), "mean_difference": float(proposed[0] - baseline[0])}
    test = stats.ttest_rel(proposed, baseline)
    return {
        "paired_t_statistic": float(test.statistic),
        "paired_t_p_value": float(test.pvalue),
        "mean_difference": float((proposed - baseline).mean()),
    }


def paired_bootstrap_difference(
    baseline_predictions: Iterable[int],
    proposed_predictions: Iterable[int],
    labels: Iterable[int],
    metric: Callable[[list[int], list[int]], float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap a proposed-minus-baseline metric difference on aligned examples."""

    baseline = np.asarray(list(baseline_predictions))
    proposed = np.asarray(list(proposed_predictions))
    expected = np.asarray(list(labels))
    if not (baseline.shape == proposed.shape == expected.shape) or expected.size == 0:
        raise ValueError("Paired bootstrap requires aligned, non-empty predictions and labels.")
    random = np.random.default_rng(seed)
    indices = np.arange(expected.size)
    observed = metric(proposed.tolist(), expected.tolist()) - metric(baseline.tolist(), expected.tolist())
    differences = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        selected = random.choice(indices, size=indices.size, replace=True)
        differences[sample_index] = metric(proposed[selected].tolist(), expected[selected].tolist()) - metric(
            baseline[selected].tolist(), expected[selected].tolist()
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be strictly between zero and one.")
    tail = (1.0 - confidence) / 2.0
    return {
        "observed_difference": float(observed),
        "bootstrap_mean_difference": float(differences.mean()),
        "confidence_level": float(confidence),
        "bootstrap_ci_lower": float(np.quantile(differences, tail)),
        "bootstrap_ci_upper": float(np.quantile(differences, 1.0 - tail)),
        "two_sided_p_value": float(2.0 * min((differences <= 0).mean(), (differences >= 0).mean())),
        "samples": float(samples),
    }
