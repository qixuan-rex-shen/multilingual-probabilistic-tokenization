"""Canonical local adapters for the downloaded GLueCoS downstream datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from .culturax_loader import normalize_text


def safe_dataset_name(repository: str) -> str:
    """Match the directory naming convention used by the download notebook."""

    return repository.replace("/", "__").replace(" ", "_")


def infer_task(repository: str) -> str:
    """Infer the configured task family from a requested GLueCoS repository."""

    upper_name = repository.upper()
    if "_NER_" in upper_name:
        return "ner"
    if "_POS_" in upper_name:
        return "pos"
    if "_SENTIMENT_" in upper_name:
        return "sentiment"
    raise ValueError(f"Cannot infer task type for {repository}")


def infer_language_pair(repository: str) -> str:
    """Return the language suffix while retaining the source repository metadata."""

    return repository.rsplit("_", maxsplit=2)[-2].lower() + "-" + repository.rsplit("_", maxsplit=2)[-1].lower()


def canonicalize_gluecos_dataset(
    raw_dataset: DatasetDict,
    repository: str,
    preprocessing: dict[str, Any],
    preserve_original_columns: bool,
) -> DatasetDict:
    """Add task-independent fields while preserving gold labels and original splits."""

    task = infer_task(repository)
    language_pair = infer_language_pair(repository)

    def add_ner_fields(example: dict[str, Any]) -> dict[str, Any]:
        tokens = example["hindi_words"]
        return {
            "tokens": tokens,
            "ner_tags": example["labels"],
            "text": " ".join(tokens),
            "task": task,
            "language_pair": language_pair,
        }

    def add_pos_fields(example: dict[str, Any]) -> dict[str, Any]:
        tokens = example["words"]
        return {
            "tokens": tokens,
            "pos_tags_primary": example["label1"],
            "pos_tags_secondary": example["label2"],
            "text": " ".join(tokens),
            "task": task,
            "language_pair": language_pair,
        }

    def add_sentiment_fields(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": normalize_text(example["text"], preprocessing),
            "sentiment_label": example["label"],
            "task": task,
            "language_pair": language_pair,
        }

    mapper = {"ner": add_ner_fields, "pos": add_pos_fields, "sentiment": add_sentiment_fields}[task]
    result = DatasetDict()
    for split_name, split in raw_dataset.items():
        remove_columns = None if preserve_original_columns else split.column_names
        mapping_identity = {
            "repository": repository,
            "split": split_name,
            "task": task,
            "language_pair": language_pair,
            "preprocessing": preprocessing,
            "preserve_original_columns": preserve_original_columns,
            "source_fingerprint": split._fingerprint,
        }
        result[split_name] = split.map(
            mapper,
            remove_columns=remove_columns,
            new_fingerprint=hashlib.sha256(
                json.dumps(mapping_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"Canonicalizing {repository}/{split_name}",
        )
    return result


def load_raw_gluecos(raw_data_root: Path, repository: str) -> DatasetDict:
    """Open a downloaded GLueCoS Dataset.save_to_disk artifact."""

    source_path = raw_data_root / "gluecos" / safe_dataset_name(repository)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Missing raw GLueCoS dataset: {source_path}")
    return load_from_disk(str(source_path))


def validate_saved_gluecos(path: Path) -> dict[str, int]:
    """Validate a processed Dataset.save_to_disk artifact."""

    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict at {path}")
    rows = {split_name: len(split) for split_name, split in dataset.items()}
    if not rows or any(count <= 0 for count in rows.values()):
        raise ValueError(f"Invalid or empty processed GLueCoS dataset: {path}")
    return rows
