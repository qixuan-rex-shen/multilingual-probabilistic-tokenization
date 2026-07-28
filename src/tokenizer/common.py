"""Shared deterministic corpus streaming for both tokenizer groups."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.data.pipeline import iter_prepared_corpus


XLMR_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def build_xlmr_single_sequence(tokenizer: Any, content_ids: list[int]) -> list[int]:
    """Wrap one content sequence with the explicit XLM-R-compatible markers.

    Some recent ``transformers`` fast-tokenizer backends expose
    ``num_special_tokens_to_add`` but no longer expose
    ``build_inputs_with_special_tokens``.  The project owns the tokenizer
    vocabulary and fixes its XLM-R token order, so constructing the single
    sequence directly is stable across those backend versions.
    """

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.sep_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define XLM-R-compatible BOS/CLS and EOS/SEP token IDs.")
    return [int(bos_token_id), *content_ids, int(eos_token_id)]


def artifact_identity(plan_path: Path, tokenizer_config: dict[str, Any], tokenizer_kind: str) -> str:
    """Hash the immutable inputs that define a tokenizer artifact."""

    payload = {
        "kind": tokenizer_kind,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "tokenizer_config": tokenizer_config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_special_tokens(configured_tokens: list[str]) -> list[str]:
    if configured_tokens != XLMR_SPECIAL_TOKENS:
        raise ValueError(
            "tokenizer.special_tokens must use the XLM-R-compatible order "
            f"{XLMR_SPECIAL_TOKENS}; received {configured_tokens}."
        )
    return configured_tokens


def iter_balanced_training_records(
    project_root: Path,
    plan_path: Path,
    tokenizer_config: dict[str, Any],
) -> Iterator[dict[str, str]]:
    """Yield one bounded, train-only, balanced record stream for both groups."""

    languages = ("en", "es", "hi")
    weights = tokenizer_config["per_language_sampling_weights"]
    if any(float(weights[language]) != 1.0 for language in languages):
        raise ValueError("The first comparison requires equal per-language tokenizer sampling weights.")
    limit = int(tokenizer_config["max_training_text_bytes_per_language"])
    used = {language: 0 for language in languages}
    finished = {language: False for language in languages}
    for record in iter_prepared_corpus(project_root, plan_path, split="train", balanced=True):
        language = record["language"]
        if language not in used or finished[language]:
            continue
        text = record["text"]
        text_bytes = len(text.encode("utf-8"))
        if limit and used[language] + text_bytes > limit:
            # The cap is an upper bound, not a requirement to consume the last
            # few bytes.  Continuing to scan every remaining shard in search
            # of a smaller document previously made a capped fit traverse the
            # full 45 GB corpus.  Freeze this language at the deterministic
            # document boundary instead; both tokenizer arms use this exact
            # stream and therefore remain matched.
            finished[language] = True
            if all(finished.values()):
                return
            continue
        used[language] += text_bytes
        yield record
        if limit and used[language] >= limit:
            finished[language] = True
            if all(finished.values()):
                return


def iter_balanced_training_text(
    project_root: Path,
    plan_path: Path,
    tokenizer_config: dict[str, Any],
) -> Iterator[str]:
    """Yield only text from the shared bounded record stream."""

    for record in iter_balanced_training_records(project_root, plan_path, tokenizer_config):
        yield record["text"]


def batched_text_iterator(text_iterator: Iterator[str], batch_documents: int) -> Iterator[list[str]]:
    """Provide bounded batches to the Rust tokenizer trainers."""

    if batch_documents <= 0:
        raise ValueError("training_batch_documents must be positive.")
    batch: list[str] = []
    for text in text_iterator:
        batch.append(text)
        if len(batch) >= batch_documents:
            yield batch
            batch = []
    if batch:
        yield batch


def write_metadata(destination: Path, metadata: dict[str, Any]) -> None:
    destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_artifact_destination(destination: Path, required_files: set[str]) -> bool:
    """Return whether a completed tokenizer artifact is present at ``destination``.

    Repository scaffolding may contain exactly one ``.gitkeep`` file.  It is
    safe to remove that empty placeholder before an atomic artifact write.  Any
    other incomplete directory is preserved and rejected so interrupted work is
    never silently overwritten.
    """

    if not destination.exists():
        return False
    if not destination.is_dir():
        raise FileExistsError(f"Tokenizer artifact destination is not a directory: {destination}")
    entry_names = {entry.name for entry in destination.iterdir()}
    if required_files.issubset(entry_names):
        return True
    if entry_names in (set(), {".gitkeep"}):
        placeholder = destination / ".gitkeep"
        print(f"Removing empty tokenizer scaffold: {destination}")
        if placeholder.is_file():
            placeholder.unlink()
        # Some synced Windows folders retain a read-only directory attribute.
        # Make this known-empty scaffold writable before removing it.
        destination.chmod(0o700)
        destination.rmdir()
        return False
    raise FileExistsError(
        f"Found an incomplete or untracked tokenizer artifact: {destination}. "
        "It has not been overwritten; inspect or archive it before retrying."
    )
