"""Aggregate paired GLueCoS results and write reproducible statistical reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score

from src.evaluation.statistics import (
    paired_bootstrap_difference,
    paired_seed_test,
    summarize_seed_scores,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    """Write a human-readable report atomically beside the JSON artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _metric_score(predictions: list[int], labels: list[int]) -> float:
    """Bootstrap the directly interpretable accuracy of aligned predictions."""

    return float(accuracy_score(labels, predictions))


def analyze_finetuning_results(
    project_root: Path,
    config: dict[str, Any],
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report seed summaries, paired seed tests, and paired prediction bootstrap.

    A bootstrap is run on the first configured seed after proving that the BPE
    and probabilistic references are identical.  The paired t-test uses every
    configured seed.  Both reports retain raw test predictions in the per-seed
    output, so later alternatives can be evaluated without retraining.
    """

    if results is None:
        source = project_root / config["paths"]["experiments"] / config["experiment"]["id"] / "downstream_metrics.json"
        if not source.is_file():
            raise FileNotFoundError(f"No fine-tuning results are available for analysis: {source}")
        results = json.loads(source.read_text(encoding="utf-8"))

    confidence = float(config["evaluation"]["confidence_level"])
    bootstrap_samples = int(config["evaluation"]["paired_bootstrap_samples"])
    seed_values = [str(int(seed)) for seed in config["evaluation"]["seeds"]]
    report: dict[str, Any] = {
        "comparison": "probabilistic_minus_bpe",
        "confidence_level": confidence,
        "significance_level": float(config["evaluation"]["significance_level"]),
        "paired_bootstrap_samples": bootstrap_samples,
        "tasks": {},
    }

    for repository, task_result in results.items():
        seeds = task_result["seeds"]
        if sorted(seeds) != sorted(seed_values):
            raise ValueError(
                f"Statistical analysis needs all configured seed results for {repository}: "
                f"expected {seed_values}, found {sorted(seeds)}"
            )
        first_seed = seed_values[0]
        baseline_first = seeds[first_seed]["bpe"]
        proposed_first = seeds[first_seed]["probabilistic"]
        if baseline_first["labels"] != proposed_first["labels"]:
            raise ValueError(
                f"Paired bootstrap references differ for {repository}/seed={first_seed}; "
                "the comparison would be invalid."
            )
        metric_names = sorted(
            set(baseline_first["metrics"]).intersection(proposed_first["metrics"]) - {"labeled_tokens"}
        )
        metric_report: dict[str, Any] = {}
        for metric in metric_names:
            baseline_scores = [float(seeds[seed]["bpe"]["metrics"][metric]) for seed in seed_values]
            proposed_scores = [float(seeds[seed]["probabilistic"]["metrics"][metric]) for seed in seed_values]
            metric_report[metric] = {
                "bpe": summarize_seed_scores(baseline_scores, confidence),
                "probabilistic": summarize_seed_scores(proposed_scores, confidence),
                "paired_seed_test": paired_seed_test(baseline_scores, proposed_scores),
            }
        report["tasks"][repository] = {
            "task": task_result["task"],
            "metrics": metric_report,
            "paired_bootstrap_accuracy": paired_bootstrap_difference(
                baseline_first["predictions"],
                proposed_first["predictions"],
                baseline_first["labels"],
                _metric_score,
                bootstrap_samples,
                int(first_seed),
                confidence,
            ),
        }

    destination = project_root / config["paths"]["results"] / "statistics" / "paired_gluecos_analysis.json"
    print(f"Saving checkpoint to: {destination}")
    _atomic_json(destination, report)
    experiment_copy = project_root / config["paths"]["experiments"] / config["experiment"]["id"] / "statistical_analysis.json"
    print(f"Saving checkpoint to: {experiment_copy}")
    _atomic_json(experiment_copy, report)
    return report


def render_experiment_summary_markdown(
    report: dict[str, Any],
    experiment_manifest: dict[str, Any] | None = None,
) -> str:
    """Render the saved paired report as a compact, auditable Markdown summary."""

    comparison_id = (experiment_manifest or {}).get("comparison_id", "unavailable")
    lines = [
        "# Multilingual Tokenization Experiment Summary",
        "",
        f"- Comparison ID: `{comparison_id}`",
        f"- Contrast: {report['comparison']}",
        f"- Confidence level: {report['confidence_level']:.0%}",
        f"- Significance threshold: p < {report['significance_level']}",
        f"- Paired bootstrap samples: {int(report['paired_bootstrap_samples'])}",
        "",
        "Positive differences favour the language-aware probabilistic candidate-fusion model.",
    ]
    for repository, task_report in report["tasks"].items():
        lines.extend(
            [
                "",
                f"## {repository}",
                "",
                f"Task: `{task_report['task']}`",
                "",
                "| Metric | BPE mean (95% CI) | Probabilistic mean (95% CI) | Mean difference | Paired seed p-value |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric_name, metric_report in task_report["metrics"].items():
            baseline = metric_report["bpe"]
            proposed = metric_report["probabilistic"]
            paired = metric_report["paired_seed_test"]
            baseline_display = (
                f"{baseline['mean']:.4f} "
                f"[{baseline['confidence_interval_lower']:.4f}, {baseline['confidence_interval_upper']:.4f}]"
            )
            proposed_display = (
                f"{proposed['mean']:.4f} "
                f"[{proposed['confidence_interval_lower']:.4f}, {proposed['confidence_interval_upper']:.4f}]"
            )
            p_value = paired["paired_t_p_value"]
            p_display = "not defined" if p_value != p_value else f"{p_value:.4g}"
            lines.append(
                f"| {metric_name} | {baseline_display} | {proposed_display} | "
                f"{paired['mean_difference']:+.4f} | {p_display} |"
            )
        bootstrap = task_report["paired_bootstrap_accuracy"]
        lines.extend(
            [
                "",
                "Paired prediction bootstrap (accuracy, first configured seed): "
                f"observed difference {bootstrap['observed_difference']:+.4f}; "
                f"{bootstrap['confidence_level']:.0%} CI "
                f"[{bootstrap['bootstrap_ci_lower']:+.4f}, {bootstrap['bootstrap_ci_upper']:+.4f}]; "
                f"two-sided p={bootstrap['two_sided_p_value']:.4g}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_experiment_summary(
    project_root: Path,
    config: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save a readable companion report for the paired JSON statistics output."""

    if report is None:
        statistics_path = project_root / config["paths"]["results"] / "statistics" / "paired_gluecos_analysis.json"
        if not statistics_path.is_file():
            raise FileNotFoundError(f"Paired statistical report is required before summarization: {statistics_path}")
        report = json.loads(statistics_path.read_text(encoding="utf-8"))
    experiment_directory = project_root / config["paths"]["experiments"] / config["experiment"]["id"]
    manifest_path = experiment_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    markdown = render_experiment_summary_markdown(report, manifest)
    result_destination = project_root / config["paths"]["results"] / "statistics" / "experiment_summary.md"
    experiment_destination = experiment_directory / "experiment_summary.md"
    for destination in (result_destination, experiment_destination):
        print(f"Saving checkpoint to: {destination}")
        _atomic_text(destination, markdown)
    return {
        "results_summary": str(result_destination),
        "experiment_summary": str(experiment_destination),
    }
