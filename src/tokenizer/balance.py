"""Post-BPE multilingual token-balance measurement for the fixed corpus stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast

from .common import iter_balanced_training_records


def measure_bpe_token_balance(
    project_root: Path,
    plan_path: Path,
    tokenizer_config: dict[str, Any],
    tokenizer_directory: Path,
    destination: Path,
) -> dict[str, Any]:
    """Measure the BPE-token distribution of the exact tokenizer-fit stream."""

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_directory))
    # This utility only counts tokenizer output; it never sends long documents to
    # a model.  Avoid Transformers' model-length warning for valid raw CulturaX
    # documents that exceed the later MLM sequence limit.
    tokenizer.model_max_length = int(1e9)
    counts = {
        language: {"documents": 0, "normalized_text_bytes": 0, "bpe_tokens": 0}
        for language in ("en", "es", "hi")
    }
    for record in iter_balanced_training_records(project_root, plan_path, tokenizer_config):
        language = record["language"]
        text = record["text"]
        counts[language]["documents"] += 1
        counts[language]["normalized_text_bytes"] += len(text.encode("utf-8"))
        counts[language]["bpe_tokens"] += len(tokenizer.encode(text, add_special_tokens=False))
    total_tokens = sum(values["bpe_tokens"] for values in counts.values())
    for values in counts.values():
        values["token_share"] = values["bpe_tokens"] / max(1, total_tokens)
    result = {
        "kind": "baseline_bpe_token_balance",
        "plan_path": str(plan_path),
        "tokenizer_directory": str(tokenizer_directory),
        "per_language": counts,
        "max_absolute_share_deviation_from_equal": max(
            abs(values["token_share"] - 1.0 / len(counts)) for values in counts.values()
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    print(f"Saving checkpoint to: {destination}")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return result
