"""Create one versionable, self-contained record for a completed experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECORD_FILENAME = "experiment_record.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> Any | None:
    return _read_json(path) if path.is_file() else None


def _read_optional_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _artifact_payload(project_root: Path, path: Path) -> dict[str, Any]:
    """Embed one generated artifact and retain a hash for auditability."""

    payload: dict[str, Any] = {
        "path": _relative_path(project_root, path),
        "present": path.is_file(),
    }
    if path.is_file():
        payload["sha256"] = _sha256_file(path)
        if path.suffix.lower() == ".json":
            payload["contents"] = _read_json(path)
        else:
            payload["contents"] = path.read_text(encoding="utf-8")
    return payload


def build_experiment_record(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Collect configuration, provenance, hardware, and every persisted result.

    The record deliberately embeds machine-readable result payloads instead of
    merely linking to ignored output directories.  It is therefore sufficient
    to inspect the final experimental state after the local artifacts have
    been cleaned up.
    """

    experiment_id = str(config["experiment"]["id"])
    experiment_directory = project_root / config["paths"]["experiments"] / experiment_id
    results_directory = project_root / config["paths"]["results"]
    artifact_paths = {
        "resolved_configuration": project_root / "configs" / "config.yaml",
        "experiment_manifest": experiment_directory / "manifest.json",
        "hardware_and_software": experiment_directory / "environment.json",
        "dataset_details": experiment_directory / "dataset_info.json",
        "tokenizer_details": experiment_directory / "tokenizer_info.json",
        "paired_model_metrics": experiment_directory / "metrics.json",
        "bpe_mlm_test": results_directory / "mlm" / "bpe_model_test_metrics.json",
        "probabilistic_mlm_test": results_directory / "mlm" / "probabilistic_test_metrics.json",
        "bpe_token_balance": results_directory / "tokenizer_balance" / "bpe_token_balance.json",
        "downstream_seed_results": experiment_directory / "downstream_metrics.json",
        "paired_statistical_analysis": results_directory / "statistics" / "paired_gluecos_analysis.json",
        "human_readable_summary": results_directory / "statistics" / "experiment_summary.md",
    }
    artifacts = {
        name: _artifact_payload(project_root, path)
        for name, path in artifact_paths.items()
    }
    missing = [name for name, artifact in artifacts.items() if not artifact["present"]]
    manifest = artifacts["experiment_manifest"].get("contents") or {}
    record: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "record_policy": {
            "destination": RECORD_FILENAME,
            "overwrite": "This file is atomically replaced whenever the final statistical-analysis stage runs.",
            "version_control": "The destination is intentionally not ignored by Git.",
        },
        "experiment": {
            "id": experiment_id,
            "comparison_id": manifest.get("comparison_id"),
            "training_seed": config.get("training", {}).get("seed"),
            "evaluation_seeds": list(config.get("evaluation", {}).get("seeds", [])),
            "configured_languages": list(config.get("data_download", {}).get("languages", [])),
            "configured_gluecos_repositories": list(config.get("data_download", {}).get("gluecos", [])),
        },
        "parameters": config,
        "artifacts": artifacts,
        "missing_artifacts": missing,
    }
    return record


def write_experiment_record(project_root: Path, config: dict[str, Any]) -> Path:
    """Atomically overwrite the tracked reproducibility record at project root."""

    destination = project_root / RECORD_FILENAME
    record = build_experiment_record(project_root, config)
    print(f"Writing consolidated experiment record to: {destination}")
    _atomic_json(destination, record)
    return destination
