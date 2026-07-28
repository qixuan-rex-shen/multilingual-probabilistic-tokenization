"""Local, self-contained experiment provenance for paired tokenizer runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _environment_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        payload["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return payload


def _tokenizer_hash(directory: Path) -> str:
    required = (directory / "tokenizer.json", directory / "training_metadata.json")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Tokenizer artifact is incomplete: {directory}")
    return _stable_hash({path.name: _sha256_file(path) for path in required})


def _experiment_directory(project_root: Path) -> Path:
    config_path = project_root / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config.get("experiment", {}).get("id", "experiment_001"))
    return project_root / config["paths"]["experiments"] / experiment_id


def prepare_paired_experiment(
    project_root: Path,
    plan_path: Path,
    group_name: str,
    tokenizer_directory: Path,
    model_settings: dict[str, Any],
    training_settings: dict[str, Any],
    group_settings: dict[str, Any] | None = None,
) -> Path:
    """Create or validate the immutable shared provenance record for a group."""

    destination = _experiment_directory(project_root)
    print(f"Saving checkpoint to: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "configs" / "config.yaml"
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (destination / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (destination / "git_commit.txt").write_text(_git_commit(project_root) + "\n", encoding="utf-8")
    _atomic_json(destination / "environment.json", _environment_payload())
    (destination / "seed.txt").write_text(str(training_settings["seed"]) + "\n", encoding="utf-8")

    source_dataset_info = project_root / "experiments" / "research_pipeline" / "dataset_info.json"
    if source_dataset_info.is_file():
        (destination / "dataset_info.json").write_bytes(source_dataset_info.read_bytes())

    corpus_hash = _sha256_file(plan_path)
    # Throughput plumbing can evolve independently of the experimental
    # condition.  Bounded CPU prefetching and ordered candidate-worker pools
    # only overlap deterministic candidate construction with GPU work; they
    # do not change corpus order, candidate paths, model inputs, optimizer
    # updates, or the effective batch.  Keep them in the saved resolved config,
    # but exclude them from the paired-comparison identity so an implementation
    # optimization does not invalidate an already-completed BPE control arm.
    comparison_training_settings = {
        key: value
        for key, value in training_settings.items()
        if key
        not in {
            "candidate_prefetch_batches",
            "candidate_preparation_workers",
            "candidate_preparation_buffer_batches",
        }
    }
    base_training = {"model": model_settings, "training": comparison_training_settings}
    base_training_hash = _stable_hash(base_training)
    tokenizer_hash = _tokenizer_hash(tokenizer_directory)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "experiment_id": destination.name,
        "created_at_utc": _utc_now(),
        "groups": {},
    }
    for key, expected in {
        "corpus_plan_sha256": corpus_hash,
        "base_training_configuration_sha256": base_training_hash,
        "random_seed": int(training_settings["seed"]),
    }.items():
        actual = manifest.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"Paired comparison mismatch for {key}: {actual!r} != {expected!r}")
        manifest[key] = expected
    manifest["comparison_id"] = _stable_hash(
        {
            "corpus": corpus_hash,
            "base_training": base_training_hash,
            "seed": manifest["random_seed"],
        }
    )[:16]
    group_payload: dict[str, Any] = {
        "tokenizer_directory": str(tokenizer_directory),
        "tokenizer_artifact_sha256": tokenizer_hash,
        "registered_at_utc": _utc_now(),
    }
    if group_settings:
        group_payload["runtime_settings"] = group_settings
    manifest["groups"][group_name] = group_payload
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        destination / "tokenizer_info.json",
        {name: details for name, details in manifest["groups"].items()},
    )
    return destination


def record_group_result(experiment_directory: Path, group_name: str, result: dict[str, Any]) -> None:
    """Persist the selected checkpoint and metrics for one paired model arm."""

    manifest_path = experiment_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if group_name not in manifest.get("groups", {}):
        raise KeyError(f"Group was not registered in paired manifest: {group_name}")
    manifest["groups"][group_name]["result"] = result
    manifest["groups"][group_name]["completed_at_utc"] = _utc_now()
    print(f"Saving checkpoint to: {experiment_directory / 'manifest.json'}")
    _atomic_json(manifest_path, manifest)
    _atomic_json(experiment_directory / "metrics.json", {name: value.get("result") for name, value in manifest["groups"].items()})
