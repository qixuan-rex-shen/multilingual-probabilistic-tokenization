"""Offline readers and deterministic preprocessing for downloaded CulturaX shards."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


_WHITESPACE_PATTERN = re.compile(r"\s+")


def load_language_metadata(language_directory: Path) -> dict[str, Any]:
    """Load the completion metadata written by the download notebook."""

    metadata_path = language_directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing CulturaX metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_language_part_paths(language_directory: Path) -> list[Path]:
    """Return every checkpointed Parquet shard in the saved order."""

    metadata = load_language_metadata(language_directory)
    if metadata.get("status") != "completed":
        raise ValueError(f"CulturaX language is not completed: {language_directory}")

    part_paths = [language_directory / part_name for part_name in metadata.get("parts", [])]
    if not part_paths:
        raise ValueError(f"No Parquet shards are listed in {language_directory / 'metadata.json'}")
    return part_paths


def validate_language_shards(language_directory: Path) -> dict[str, int]:
    """Validate all Parquet footers without reading the full text payload."""

    metadata = load_language_metadata(language_directory)
    rows = 0
    disk_bytes = 0
    part_paths = resolve_language_part_paths(language_directory)
    for part_path in part_paths:
        parquet_file = pq.ParquetFile(part_path)
        if parquet_file.metadata.num_rows <= 0:
            raise ValueError(f"Empty CulturaX shard: {part_path}")
        if parquet_file.schema_arrow.names != ["text", "language"]:
            raise ValueError(f"Unexpected CulturaX schema in {part_path}: {parquet_file.schema_arrow}")
        rows += parquet_file.metadata.num_rows
        disk_bytes += part_path.stat().st_size

    expected_rows = int(metadata.get("rows", -1))
    if rows != expected_rows:
        raise ValueError(f"Row-count mismatch in {language_directory}: {rows} != {expected_rows}")

    return {
        "rows": rows,
        "parts": len(part_paths),
        "disk_bytes": disk_bytes,
        "text_bytes": int(metadata["actual_text_bytes"]),
    }


def normalize_text(text: str, preprocessing: dict[str, Any]) -> str:
    """Apply the configured text normalization exactly once at iteration time."""

    normalization = preprocessing.get("unicode_normalization")
    if normalization:
        text = unicodedata.normalize(normalization, text)
    if preprocessing.get("collapse_whitespace", False):
        text = _WHITESPACE_PATTERN.sub(" ", text)
    if preprocessing.get("strip_whitespace", False):
        text = text.strip()
    return text


def text_passes_filters(text: str, filtering: dict[str, Any]) -> bool:
    """Return whether normalized text meets configured character-length limits."""

    return int(filtering["min_characters"]) <= len(text) <= int(filtering["max_characters"])


def stable_split(
    text: str,
    language: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    """Assign a record deterministically without storing a separate split index."""

    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    payload = f"{seed}\0{language}\0{text}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    train_fraction = 1.0 - validation_fraction - test_fraction
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def iter_language_records(
    language_directory: Path,
    language: str,
    preprocessing: dict[str, Any],
    filtering: dict[str, Any],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    split: str | None = None,
    batch_rows: int = 10_000,
) -> Iterator[dict[str, str]]:
    """Yield normalized local CulturaX records from finalized shards only.

    The raw corpus remains immutable. This function is deliberately streaming so
    tokenizer and MLM stages do not need a duplicate uncompressed 45 GB corpus.
    """

    if split not in (None, "train", "validation", "test"):
        raise ValueError("split must be None, 'train', 'validation', or 'test'.")

    for part_path in resolve_language_part_paths(language_directory):
        parquet_file = pq.ParquetFile(part_path)
        for batch in parquet_file.iter_batches(columns=["text", "language"], batch_size=batch_rows):
            columns = batch.to_pydict()
            for raw_text, raw_language in zip(columns["text"], columns["language"], strict=True):
                if raw_language != language:
                    raise ValueError(f"Language mismatch in {part_path}: {raw_language!r} != {language!r}")
                text = normalize_text(raw_text, preprocessing)
                if not text_passes_filters(text, filtering):
                    continue
                assigned_split = stable_split(text, language, seed, validation_fraction, test_fraction)
                if split is None or assigned_split == split:
                    yield {"text": text, "language": language}


def source_signature(language_directory: Path) -> str:
    """Create a stable identity for a downloaded language corpus."""

    metadata_path = language_directory / "metadata.json"
    return hashlib.sha256(metadata_path.read_bytes()).hexdigest()


def round_robin_language_records(language_iterables: dict[str, Iterable[dict[str, str]]]) -> Iterator[dict[str, str]]:
    """Interleave language streams for provisional multilingual balancing."""

    iterators = {language: iter(records) for language, records in language_iterables.items()}
    while iterators:
        for language in list(iterators):
            try:
                yield next(iterators[language])
            except StopIteration:
                del iterators[language]
