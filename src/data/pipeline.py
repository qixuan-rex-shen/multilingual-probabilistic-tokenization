"""Checkpointed, local-only data preparation for the research pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from .culturax_loader import (
    iter_language_records,
    load_language_metadata,
    resolve_language_part_paths,
    round_robin_language_records,
    source_signature,
    validate_language_shards,
)
from .gluecos_loader import (
    canonicalize_gluecos_dataset,
    load_raw_gluecos,
    safe_dataset_name,
    validate_saved_gluecos,
)


PIPELINE_MANIFEST_VERSION = 1
CORPUS_PLAN_FORMAT = "culturax_training_manifest/v1"


def utc_now() -> str:
    """Return an unambiguous timestamp for manifests."""

    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so an interrupted notebook cannot corrupt a manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _configuration_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_to_root(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _load_pipeline_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manifest_version") != PIPELINE_MANIFEST_VERSION:
            raise ValueError(f"Unsupported research-pipeline manifest: {manifest_path}")
        manifest.setdefault("stages", {})
        return manifest
    return {
        "manifest_version": PIPELINE_MANIFEST_VERSION,
        "created_at_utc": utc_now(),
        "stages": {},
    }


def _save_pipeline_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def _stage_is_reusable(
    manifest: dict[str, Any],
    stage_id: str,
    signature: str,
    validator: Callable[[], bool],
) -> bool:
    stage = manifest["stages"].get(stage_id, {})
    if stage.get("status") != "completed" or stage.get("signature") != signature:
        return False
    try:
        return validator()
    except Exception:
        return False


def _mark_stage_running(
    manifest_path: Path,
    manifest: dict[str, Any],
    stage_id: str,
    signature: str,
    destination: Path | None = None,
) -> None:
    manifest["stages"][stage_id] = {
        "status": "running",
        "signature": signature,
        "started_at_utc": utc_now(),
        **({"destination": str(destination)} if destination is not None else {}),
    }
    _save_pipeline_manifest(manifest_path, manifest)


def _mark_stage_completed(
    manifest_path: Path,
    manifest: dict[str, Any],
    stage_id: str,
    details: dict[str, Any],
) -> None:
    manifest["stages"][stage_id].update(
        {"status": "completed", "completed_at_utc": utc_now(), "details": details}
    )
    _save_pipeline_manifest(manifest_path, manifest)


def _mark_stage_failed(
    manifest_path: Path,
    manifest: dict[str, Any],
    stage_id: str,
    error: Exception,
) -> None:
    manifest["stages"][stage_id].update(
        {
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        }
    )
    _save_pipeline_manifest(manifest_path, manifest)


def _pipeline_paths(project_root: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = config["paths"]
    raw_root = project_root / paths["raw_data"]
    processed_root = project_root / paths["processed_data"]
    manifest_path = project_root / config["checkpointing"]["manifest_path"]
    return raw_root, processed_root, manifest_path


def _validate_preparation_configuration(config: dict[str, Any]) -> None:
    """Reject settings that this first local preparation implementation cannot honor."""

    preparation = config["data_preparation"]
    if preparation["corpus_storage"]["mode"] != "manifest_only":
        raise ValueError("Only data_preparation.corpus_storage.mode: manifest_only is supported.")
    if preparation["splits"]["hash_algorithm"].lower() != "sha256":
        raise ValueError("Only data_preparation.splits.hash_algorithm: sha256 is supported.")
    train_fraction = float(preparation["splits"]["train_fraction"])
    validation_fraction = float(preparation["splits"]["validation_fraction"])
    test_fraction = float(preparation["splits"]["test_fraction"])
    if any(fraction < 0.0 or fraction >= 1.0 for fraction in (train_fraction, validation_fraction, test_fraction)):
        raise ValueError("All data_preparation split fractions must be in [0, 1).")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("data_preparation train/validation/test fractions must sum to 1.")
    if int(preparation["filtering"]["min_characters"]) < 0:
        raise ValueError("data_preparation.filtering.min_characters must be non-negative.")
    if int(preparation["filtering"]["max_characters"]) < int(preparation["filtering"]["min_characters"]):
        raise ValueError("data_preparation.filtering.max_characters must be at least min_characters.")
    if preparation["preprocessing"]["unicode_normalization"] != config["tokenizer"]["normalization"]:
        raise ValueError(
            "Tokenizer and data-preparation Unicode normalization must match for a fair comparison."
        )


def validate_raw_downloads(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate every raw CulturaX shard and GLueCoS DatasetDict before reuse."""

    raw_root, _, _ = _pipeline_paths(project_root, config)
    languages = config["data_download"]["languages"]
    culturax_summary: dict[str, dict[str, int]] = {}
    for language in languages:
        culturax_summary[language] = validate_language_shards(raw_root / "culturax" / language)

    tolerance = int(config["data_preparation"]["balancing"]["source_byte_tolerance"])
    text_bytes = [summary["text_bytes"] for summary in culturax_summary.values()]
    if max(text_bytes) - min(text_bytes) > tolerance:
        raise ValueError(
            "CulturaX languages are not source-byte balanced within the configured tolerance: "
            f"{culturax_summary}"
        )

    gluecos_summary: dict[str, dict[str, int]] = {}
    for repository in config["data_download"]["gluecos"]:
        raw_dataset = load_raw_gluecos(raw_root, repository)
        split_rows = {split_name: len(split) for split_name, split in raw_dataset.items()}
        if not split_rows or any(rows <= 0 for rows in split_rows.values()):
            raise ValueError(f"Invalid raw GLueCoS dataset: {repository}")
        gluecos_summary[repository] = split_rows

    return {
        "raw_data_root": _relative_to_root(project_root, raw_root),
        "culturax": culturax_summary,
        "gluecos": gluecos_summary,
        "source_byte_difference": max(text_bytes) - min(text_bytes),
    }


def _build_corpus_plan(project_root: Path, config: dict[str, Any], purpose: str) -> dict[str, Any]:
    raw_root, _, _ = _pipeline_paths(project_root, config)
    preparation = config["data_preparation"]
    languages = []
    for language in config["data_download"]["languages"]:
        language_directory = raw_root / "culturax" / language
        metadata = load_language_metadata(language_directory)
        languages.append(
            {
                "language": language,
                "source_directory": _relative_to_root(project_root, language_directory),
                "source_metadata_sha256": source_signature(language_directory),
                "part_paths": [
                    _relative_to_root(project_root, part_path)
                    for part_path in resolve_language_part_paths(language_directory)
                ],
                "source_rows": int(metadata["rows"]),
                "source_text_bytes": int(metadata["actual_text_bytes"]),
            }
        )

    plan = {
        "format": CORPUS_PLAN_FORMAT,
        "manifest_version": int(preparation["manifest_version"]),
        "purpose": purpose,
        "created_at_utc": utc_now(),
        "languages": languages,
        "preprocessing": preparation["preprocessing"],
        "filtering": preparation["filtering"],
        "splits": {
            "seed": int(config["training"]["seed"]),
            **preparation["splits"],
        },
        "streaming": {
            "parquet_batch_rows": int(preparation["corpus_storage"]["parquet_batch_rows"]),
            "storage_mode": preparation["corpus_storage"]["mode"],
        },
        "balancing": {
            **preparation["balancing"],
            "status": "source_text_bytes_balanced",
            "final_token_count_status": "pending_bpe_tokenizer_training",
        },
    }
    identity = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at_utc"}
    }
    plan["identity_sha256"] = _configuration_fingerprint(identity)
    return plan


def _validate_corpus_plan(plan_path: Path, project_root: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format") != CORPUS_PLAN_FORMAT:
        raise ValueError(f"Unexpected corpus plan format: {plan_path}")
    for language_entry in plan["languages"]:
        source_directory = project_root / language_entry["source_directory"]
        if source_signature(source_directory) != language_entry["source_metadata_sha256"]:
            raise ValueError(f"Raw source changed since plan creation: {source_directory}")
        for relative_part_path in language_entry["part_paths"]:
            if not (project_root / relative_part_path).is_file():
                raise FileNotFoundError(f"Missing planned CulturaX shard: {relative_part_path}")
    return plan


def _write_or_reuse_corpus_plan(plan_path: Path, plan: dict[str, Any], project_root: Path) -> dict[str, Any]:
    if plan_path.is_file():
        existing_plan = _validate_corpus_plan(plan_path, project_root)
        if existing_plan.get("identity_sha256") != plan["identity_sha256"]:
            raise FileExistsError(
                f"Existing corpus plan has a different configuration: {plan_path}. "
                "Create a new processed-data location or explicitly archive the old plan."
            )
        return existing_plan

    print(f"Saving checkpoint to: {plan_path}")
    atomic_write_json(plan_path, plan)
    return _validate_corpus_plan(plan_path, project_root)


def prepare_culturax_training_manifests(project_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Create local, immutable manifests for tokenizer and MLM corpus streaming."""

    _, processed_root, manifest_path = _pipeline_paths(project_root, config)
    manifest = _load_pipeline_manifest(manifest_path)
    outcomes: dict[str, dict[str, Any]] = {}
    for purpose in ("tokenizer_training", "mlm_training"):
        plan_path = processed_root / purpose / "manifest.json"
        plan = _build_corpus_plan(project_root, config, purpose)
        stage_id = f"culturax_plan/{purpose}"
        if _stage_is_reusable(
            manifest,
            stage_id,
            plan["identity_sha256"],
            lambda path=plan_path: _validate_corpus_plan(path, project_root) is not None,
        ):
            outcomes[purpose] = _validate_corpus_plan(plan_path, project_root)
            continue
        _mark_stage_running(manifest_path, manifest, stage_id, plan["identity_sha256"], plan_path)
        try:
            outcomes[purpose] = _write_or_reuse_corpus_plan(plan_path, plan, project_root)
            _mark_stage_completed(
                manifest_path,
                manifest,
                stage_id,
                {
                    "plan_path": _relative_to_root(project_root, plan_path),
                    "source_text_bytes": sum(item["source_text_bytes"] for item in plan["languages"]),
                    "languages": [item["language"] for item in plan["languages"]],
                },
            )
        except Exception as error:
            _mark_stage_failed(manifest_path, manifest, stage_id, error)
            raise
    return outcomes


def prepare_downstream_datasets(project_root: Path, config: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Canonicalize and checkpoint the small downstream datasets locally."""

    raw_root, processed_root, manifest_path = _pipeline_paths(project_root, config)
    manifest = _load_pipeline_manifest(manifest_path)
    preparation = config["data_preparation"]
    outcomes: dict[str, dict[str, int]] = {}

    for repository in config["data_download"]["gluecos"]:
        dataset_name = safe_dataset_name(repository)
        destination = processed_root / "downstream_tasks" / dataset_name
        stage_id = f"downstream/{dataset_name}"
        signature = _configuration_fingerprint(
            {
                "repository": repository,
                "preprocessing": preparation["preprocessing"],
                "preserve_original_columns": preparation["downstream"]["preserve_original_columns"],
            }
        )
        if _stage_is_reusable(
            manifest,
            stage_id,
            signature,
            lambda path=destination: validate_saved_gluecos(path) is not None,
        ):
            outcomes[repository] = validate_saved_gluecos(destination)
            continue
        if destination.exists():
            raise FileExistsError(
                f"Found an untracked or differently configured processed dataset: {destination}. "
                "This notebook will not overwrite it."
            )

        temporary_destination = destination.with_name(f"{destination.name}.incomplete")
        if temporary_destination.exists():
            raise FileExistsError(
                f"Found an incomplete processed dataset: {temporary_destination}. "
                "Inspect it before retrying; this notebook will not overwrite it."
            )

        _mark_stage_running(manifest_path, manifest, stage_id, signature, destination)
        try:
            canonical_dataset = canonicalize_gluecos_dataset(
                load_raw_gluecos(raw_root, repository),
                repository,
                preparation["preprocessing"],
                bool(preparation["downstream"]["preserve_original_columns"]),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving checkpoint to: {temporary_destination}")
            canonical_dataset.save_to_disk(str(temporary_destination))
            outcomes[repository] = validate_saved_gluecos(temporary_destination)
            temporary_destination.replace(destination)
            _mark_stage_completed(
                manifest_path,
                manifest,
                stage_id,
                {"destination": _relative_to_root(project_root, destination), "split_rows": outcomes[repository]},
            )
        except Exception as error:
            _mark_stage_failed(manifest_path, manifest, stage_id, error)
            raise
    return outcomes


def iter_prepared_corpus(
    project_root: Path,
    plan_path: Path,
    split: str | None = None,
    balanced: bool = False,
) -> Iterator[dict[str, str]]:
    """Stream normalized text from a saved plan for tokenizer or MLM code."""

    plan = _validate_corpus_plan(plan_path, project_root)
    streams = {
        entry["language"]: iter_language_records(
            project_root / entry["source_directory"],
            entry["language"],
            plan["preprocessing"],
            plan["filtering"],
            int(plan["splits"]["seed"]),
            float(plan["splits"]["validation_fraction"]),
            float(plan["splits"]["test_fraction"]),
            split=split,
            batch_rows=int(plan["streaming"]["parquet_batch_rows"]),
        )
        for entry in plan["languages"]
    }
    if balanced:
        yield from round_robin_language_records(streams)
    else:
        for stream in streams.values():
            yield from stream


def write_pipeline_dataset_info(
    project_root: Path,
    config: dict[str, Any],
    raw_summary: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    downstream_summary: dict[str, dict[str, int]],
) -> Path:
    """Record input provenance and preparation choices for later experiments."""

    _, _, manifest_path = _pipeline_paths(project_root, config)
    experiment_directory = manifest_path.parent
    dataset_info_path = experiment_directory / "dataset_info.json"
    payload = {
        "created_at_utc": utc_now(),
        "raw_validation": raw_summary,
        "culturax_plans": {
            purpose: {
                "identity_sha256": plan["identity_sha256"],
                "purpose": plan["purpose"],
                "source_text_bytes": sum(entry["source_text_bytes"] for entry in plan["languages"]),
                "balancing": plan["balancing"],
            }
            for purpose, plan in plans.items()
        },
        "downstream_split_rows": downstream_summary,
    }
    print(f"Saving checkpoint to: {dataset_info_path}")
    atomic_write_json(dataset_info_path, payload)
    return dataset_info_path


def save_resolved_configuration(project_root: Path, config: dict[str, Any]) -> Path:
    """Save an immutable configuration snapshot keyed by its content hash."""

    _, _, manifest_path = _pipeline_paths(project_root, config)
    fingerprint = _configuration_fingerprint(config)
    snapshot_path = manifest_path.parent / f"config_{fingerprint[:12]}.yaml"
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to: {snapshot_path}")
        snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return snapshot_path


def run_data_preparation(
    project_root: Path,
    config: dict[str, Any],
    include_downstream: bool = False,
) -> dict[str, Any]:
    """Execute the data-only stage of the local research pipeline.

    `train` creates missing artifacts, `resume` validates/reuses completed stages,
    and `evaluate` performs validation only without writing new data artifacts.
    """

    mode = config["runtime"]["mode"]
    if mode not in {"train", "resume", "evaluate"}:
        raise ValueError(f"Unsupported runtime.mode: {mode}")
    _validate_preparation_configuration(config)
    _, _, manifest_path = _pipeline_paths(project_root, config)
    manifest = _load_pipeline_manifest(manifest_path)
    raw_signature = _configuration_fingerprint(
        {"data_download": config["data_download"], "data_preparation": config["data_preparation"]}
    )
    stage_id = "raw_validation"
    if mode == "evaluate":
        raw_summary = validate_raw_downloads(project_root, config)
    else:
        _mark_stage_running(manifest_path, manifest, stage_id, raw_signature)
        try:
            raw_summary = validate_raw_downloads(project_root, config)
            _mark_stage_completed(manifest_path, manifest, stage_id, raw_summary)
        except Exception as error:
            _mark_stage_failed(manifest_path, manifest, stage_id, error)
            raise

    if mode == "evaluate":
        _, processed_root, _ = _pipeline_paths(project_root, config)
        plans = {
            purpose: _validate_corpus_plan(processed_root / purpose / "manifest.json", project_root)
            for purpose in ("tokenizer_training", "mlm_training")
        }
        downstream_summary = (
            {
                repository: validate_saved_gluecos(processed_root / "downstream_tasks" / safe_dataset_name(repository))
                for repository in config["data_download"]["gluecos"]
            }
            if include_downstream
            else {}
        )
    else:
        save_resolved_configuration(project_root, config)
        plans = prepare_culturax_training_manifests(project_root, config)
        downstream_summary = prepare_downstream_datasets(project_root, config) if include_downstream else {}

    dataset_info_path = manifest_path.parent / "dataset_info.json"
    if mode != "evaluate":
        dataset_info_path = write_pipeline_dataset_info(
            project_root, config, raw_summary, plans, downstream_summary
        )
    return {
        "runtime_mode": mode,
        "raw_summary": raw_summary,
        "plans": {
            purpose: str((_pipeline_paths(project_root, config)[1] / purpose / "manifest.json"))
            for purpose in plans
        },
        "downstream_summary": downstream_summary,
        "dataset_info_path": str(dataset_info_path),
        "pipeline_manifest_path": str(manifest_path),
    }
