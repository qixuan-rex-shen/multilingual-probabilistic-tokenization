"""Saved validation diagnostics for the matched tokenizer artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerFast

from src.data.pipeline import iter_prepared_corpus
from src.language.language_classifier import CulturaXLanguagePriorClassifier

from .common import build_xlmr_single_sequence
from .probabilistic import UnigramCandidateTokenizer


def validate_candidate_character_alignment(candidates: list[Any]) -> None:
    """Ensure each candidate's non-special offsets are valid original spans."""

    for candidate in candidates:
        previous_start = -1
        if len(candidate.input_ids) != len(candidate.offsets):
            raise ValueError("Candidate IDs and character offsets have different lengths.")
        for start, end in candidate.offsets:
            if start < 0 and end < 0:
                continue
            # NFKC can expand one original character into several normalized
            # characters.  Adjacent candidate pieces may then legitimately
            # project to overlapping source spans; their starts must still be
            # monotonic in original-text order.
            if start < 0 or end <= start or start < previous_start:
                raise ValueError(f"Invalid non-monotonic candidate character span: {(start, end)}")
            previous_start = start


def run_tokenizer_diagnostics(
    project_root: Path,
    plan_path: Path,
    bpe_directory: Path,
    probabilistic_directory: Path,
    language_classifier_directory: Path,
    max_sequence_length: int,
    examples_per_language: int,
    destination: Path,
) -> dict[str, Any]:
    """Compare frozen tokenizer behavior on a bounded CulturaX validation sample."""

    bpe = PreTrainedTokenizerFast.from_pretrained(str(bpe_directory))
    probabilistic = UnigramCandidateTokenizer.from_pretrained(probabilistic_directory)
    language_classifier = CulturaXLanguagePriorClassifier.load(language_classifier_directory)
    if examples_per_language <= 0:
        raise ValueError("examples_per_language must be positive.")
    metrics: dict[str, dict[str, float]] = {
        group: defaultdict(float) for group in ("bpe", "probabilistic")
    }
    seen = {language: 0 for language in ("en", "es", "hi")}
    for record in iter_prepared_corpus(project_root, plan_path, split="validation", balanced=True):
        language = record["language"]
        if seen[language] >= examples_per_language:
            continue
        text = record["text"]
        bpe_ids = bpe.encode(text, add_special_tokens=False)
        probabilities = language_classifier.predict_probabilities(text)
        candidates = probabilistic.encode_candidates(
            text, max_sequence_length=max_sequence_length, language_probabilities=probabilities
        )
        if not candidates:
            raise ValueError(f"No probabilistic candidate produced for validation text in {language}.")
        validate_candidate_character_alignment(candidates)
        selected = candidates[0]
        metrics["bpe"]["documents"] += 1
        metrics["bpe"]["tokens"] += len(bpe_ids)
        metrics["bpe"]["characters"] += len(text)
        metrics["bpe"]["unknown_tokens"] += sum(token_id == bpe.unk_token_id for token_id in bpe_ids)
        metrics["probabilistic"]["documents"] += 1
        metrics["probabilistic"]["tokens"] += len(selected.input_ids) - len(
            build_xlmr_single_sequence(probabilistic.tokenizer, [])
        )
        metrics["probabilistic"]["characters"] += len(text)
        metrics["probabilistic"]["candidate_count"] += len(candidates)
        metrics["probabilistic"]["top_candidate_probability"] += selected.prior_probability
        seen[language] += 1
        if all(count >= examples_per_language for count in seen.values()):
            break
    if any(count < examples_per_language for count in seen.values()):
        raise ValueError(f"Insufficient validation examples for diagnostics: {seen}")
    summary: dict[str, dict[str, float]] = {}
    for group, values in metrics.items():
        documents = values["documents"]
        summary[group] = {
            "documents": documents,
            "tokens": values["tokens"],
            "characters": values["characters"],
            "tokens_per_document": values["tokens"] / documents,
            "characters_per_token": values["characters"] / max(1.0, values["tokens"]),
            "unknown_rate": values.get("unknown_tokens", 0.0) / max(1.0, values["tokens"]),
        }
        if group == "probabilistic":
            summary[group]["mean_candidate_count"] = values["candidate_count"] / documents
            summary[group]["mean_top_candidate_probability"] = values["top_candidate_probability"] / documents
    result = {
        "kind": "tokenizer_validation_diagnostics",
        "plan_path": str(plan_path),
        "examples_per_language": examples_per_language,
        "languages": seen,
        "alignment_validated": True,
        "metrics": summary,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return result
