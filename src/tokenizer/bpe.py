"""From-scratch BPE control tokenizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from .common import (
    artifact_identity,
    batched_text_iterator,
    iter_balanced_training_text,
    prepare_artifact_destination,
    validate_special_tokens,
    write_metadata,
)


def _build_bpe_tokenizer(config: dict[str, Any]) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _as_transformers_tokenizer(tokenizer: Tokenizer, max_sequence_length: int) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=max_sequence_length,
    )


def validate_bpe_artifact(destination: Path, expected_identity: str | None = None) -> dict[str, Any]:
    metadata_path = destination / "training_metadata.json"
    tokenizer_path = destination / "tokenizer.json"
    if not metadata_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"Incomplete BPE artifact: {destination}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "bpe" or (expected_identity and metadata.get("identity_sha256") != expected_identity):
        raise ValueError(f"BPE artifact metadata does not match: {destination}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(destination))
    if len(tokenizer) != int(metadata["vocab_size"]):
        raise ValueError(f"BPE vocabulary mismatch in {destination}")
    return metadata


def train_bpe_tokenizer(
    project_root: Path,
    plan_path: Path,
    tokenizer_config: dict[str, Any],
    max_sequence_length: int,
    destination: Path,
) -> dict[str, Any]:
    """Train and atomically save the matched BPE control artifact."""

    special_tokens = validate_special_tokens(list(tokenizer_config["special_tokens"]))
    identity = artifact_identity(plan_path, tokenizer_config, "bpe")
    if prepare_artifact_destination(destination, {"tokenizer.json", "training_metadata.json"}):
        return validate_bpe_artifact(destination, identity)
    temporary_destination = destination.with_name(f"{destination.name}.incomplete")
    if temporary_destination.exists():
        raise FileExistsError(f"Incomplete BPE tokenizer artifact exists: {temporary_destination}")

    tokenizer = _build_bpe_tokenizer(tokenizer_config)
    trainer = trainers.BpeTrainer(
        vocab_size=int(tokenizer_config["vocab_size"]),
        min_frequency=int(tokenizer_config["min_frequency"]),
        special_tokens=special_tokens,
    )
    text_stream = batched_text_iterator(
        iter_balanced_training_text(project_root, plan_path, tokenizer_config),
        int(tokenizer_config["training_batch_documents"]),
    )
    tokenizer.train_from_iterator(text_stream, trainer=trainer)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination.mkdir(parents=True, exist_ok=False)
    print(f"Saving checkpoint to: {temporary_destination}")
    fast_tokenizer = _as_transformers_tokenizer(tokenizer, max_sequence_length)
    fast_tokenizer.save_pretrained(str(temporary_destination))
    metadata = {
        "kind": "bpe",
        "identity_sha256": identity,
        "vocab_size": len(fast_tokenizer),
        "plan_path": str(plan_path),
        "tokenizer_config": tokenizer_config,
    }
    write_metadata(temporary_destination / "training_metadata.json", metadata)
    temporary_destination.replace(destination)
    return validate_bpe_artifact(destination, identity)
