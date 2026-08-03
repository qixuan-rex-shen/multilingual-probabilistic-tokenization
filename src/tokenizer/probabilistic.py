"""Language-aware Unigram tokenizer training for the proposed comparison group."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from .common import (
    artifact_identity,
    batched_text_iterator,
    build_xlmr_single_sequence,
    iter_balanced_training_text,
    prepare_artifact_destination,
    validate_special_tokens,
    write_metadata,
)


@dataclass(frozen=True)
class CandidateTokenization:
    """One candidate path and its original-character offset mapping."""

    input_ids: list[int]
    offsets: list[tuple[int, int]]
    token_score: float
    prior_probability: float
    language_evidence: list[float]


def _normalize_with_original_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return NFKC text and an explicit normalized-character-to-source map.

    Tokenizer-library offsets refer to the user-provided string, while the
    dynamic-programming Unigram search operates over normalized vocabulary
    pieces.  For ordinary text this is a one-to-one map.  NFKC can expand or
    compose characters, so replacement blocks deliberately map to their full
    original span; projected token spans may therefore overlap, which is the
    faithful representation of a many-to-one normalization.
    """

    normalized = unicodedata.normalize("NFKC", text)
    if normalized == text:
        return normalized, [(index, index + 1) for index in range(len(text))]
    if not normalized:
        return normalized, []

    spans: list[tuple[int, int] | None] = [None] * len(normalized)
    matcher = SequenceMatcher(a=text, b=normalized, autojunk=False)
    for tag, original_start, original_end, normalized_start, normalized_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(normalized_end - normalized_start):
                spans[normalized_start + offset] = (original_start + offset, original_start + offset + 1)
            continue
        if normalized_start == normalized_end:
            continue
        # NFKC replacements normally consume at least one source character.
        # For the rare insertion-like case, attach the result to the nearest
        # real source character so every non-special token retains a valid span.
        if original_start == original_end:
            original_start = max(0, min(original_start, len(text) - 1))
            original_end = min(len(text), original_start + 1)
        for index in range(normalized_start, normalized_end):
            spans[index] = (original_start, original_end)

    fallback = (0, min(1, len(text)))
    return normalized, [span if span is not None else fallback for span in spans]


def _original_span(
    normalized_character_spans: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Project a half-open normalized span back to its source-text envelope."""

    selected = normalized_character_spans[start:end]
    if not selected:
        return (-1, -1)
    return min(span_start for span_start, _ in selected), max(span_end for _, span_end in selected)


class UnigramCandidateTokenizer:
    """Recover top Unigram segmentations from a frozen tokenizer artifact.

    The class is intentionally independent of the neural fusion module.  It
    produces token IDs, original-text spans, Unigram log scores, and lightweight
    candidate language evidence.  A caller may supply ``language_probabilities``
    when selecting tokenizer candidates; during MLM, the differentiable router
    computes its own probabilities from candidate embeddings and combines them
    in ``LanguageConditionedCandidateFusion``.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, tokenizer_json: dict[str, Any], settings: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.settings = settings
        vocabulary = tokenizer.get_vocab()
        # These IDs and the vocabulary size are immutable properties of the
        # frozen artifact.  Keeping Python integer copies avoids repeatedly
        # crossing the native ``tokenizers`` boundary from the long-running
        # training process while leaving every encoded candidate unchanged.
        self.vocab_size = len(vocabulary)
        self.pad_token_id = self._required_token_id(tokenizer.pad_token_id, "pad")
        self.mask_token_id = self._required_token_id(tokenizer.mask_token_id, "mask")
        self.bos_token_id = self._required_token_id(
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id,
            "BOS/CLS",
        )
        self.eos_token_id = self._required_token_id(
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id,
            "EOS/SEP",
        )
        self.special_token_ids = tuple(int(token_id) for token_id in tokenizer.all_special_ids)
        self.special_token_count = 2
        raw_vocab = tokenizer_json.get("model", {}).get("vocab", [])
        self.pieces_by_first_character: dict[str, list[tuple[str, int, float]]] = {}
        if isinstance(raw_vocab, list):
            for entry in raw_vocab:
                if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                    continue
                piece, score = entry
                token_id = vocabulary.get(piece)
                if token_id is None or not piece or piece in tokenizer.all_special_tokens:
                    continue
                if any(character.isspace() for character in piece):
                    continue
                self.pieces_by_first_character.setdefault(piece[0], []).append((piece, token_id, float(score)))
        for pieces in self.pieces_by_first_character.values():
            pieces.sort(key=lambda item: (-len(item[0]), -item[2], item[0]))
        # The original per-character bucket remains available for transparent
        # inspection.  A prefix trie avoids testing every vocabulary piece with
        # ``str.startswith`` at every word position during top-k decoding.
        self.piece_tries: dict[str, dict[str, Any]] = {}
        for first_character, pieces in self.pieces_by_first_character.items():
            root: dict[str, Any] = {}
            for piece, token_id, piece_score in pieces:
                node = root
                for character in piece:
                    node = node.setdefault(character, {})
                node.setdefault("_pieces", []).append((piece, token_id, piece_score))
            self.piece_tries[first_character] = root

    @staticmethod
    def _required_token_id(token_id: int | None, name: str) -> int:
        if token_id is None:
            raise ValueError(f"Frozen tokenizer is missing its {name} token ID.")
        return int(token_id)

    def configure_candidate_selection(self, runtime_settings: dict[str, Any]) -> dict[str, Any]:
        """Apply a recorded runtime top-k selection policy to this instance.

        The frozen Unigram vocabulary and token scores are never altered here.
        These settings only control how many paths are retained and how their
        scores are temperature-normalized during MLM/downstream inference.
        """

        keys = ("temperature", "max_candidates", "min_probability", "language_threshold", "alpha", "beta")
        selected = {key: runtime_settings[key] for key in keys if key in runtime_settings}
        updated = {**self.settings, **selected}
        if float(updated["temperature"]) <= 0:
            raise ValueError("temperature must be positive.")
        if int(updated["max_candidates"]) <= 0:
            raise ValueError("max_candidates must be positive.")
        if not 0.0 <= float(updated["min_probability"]) <= 1.0:
            raise ValueError("min_probability must be in [0, 1].")
        if not 0.0 <= float(updated["language_threshold"]) <= 1.0:
            raise ValueError("language_threshold must be in [0, 1].")
        self.settings = updated
        return {key: self.settings[key] for key in keys}

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "UnigramCandidateTokenizer":
        source = Path(directory)
        metadata = validate_probabilistic_artifact(source)
        tokenizer_json = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(source))
        return cls(tokenizer, tokenizer_json, dict(metadata["tokenizer_config"]))

    def _word_paths(self, word: str, normalized_start: int, top_k: int) -> list[tuple[float, list[tuple[int, int, int]]]]:
        """Use a small dynamic-programming beam over a normalized word."""

        states: list[list[tuple[float, list[tuple[int, int, int]]]]] = [[] for _ in range(len(word) + 1)]
        states[0] = [(0.0, [])]
        for start in range(len(word)):
            if not states[start]:
                continue
            matches = self._matching_pieces(word, start)
            touched_ends: set[int] = set()
            for score, path in states[start]:
                for piece, token_id, piece_score in matches:
                    end = start + len(piece)
                    if end <= len(word) and word.startswith(piece, start):
                        states[end].append(
                            (score + piece_score, path + [(token_id, normalized_start + start, normalized_start + end)])
                        )
                        touched_ends.add(end)
            # A state can only grow when it is a destination of the current
            # expansion.  The old code rechecked every future state after each
            # character, causing tens of millions of redundant ``len`` calls
            # on normal 512-token documents.  Trimming only changed states is
            # exactly equivalent: every unchanged state was already bounded
            # immediately after the expansion that last modified it.
            for end in touched_ends:
                if len(states[end]) > top_k * 8:
                    states[end] = sorted(states[end], key=lambda value: value[0], reverse=True)[: top_k * 4]
        paths = sorted(states[-1], key=lambda value: value[0], reverse=True)[:top_k]
        if paths:
            return paths
        # Frozen fast-tokenizer fallback keeps the pipeline usable for unusual
        # Unicode pieces while preserving the real original-character offsets.
        encoded = self.tokenizer(word, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        if not ids:
            return []
        fallback_score = -20.0 * len(ids)
        return [
            (
                fallback_score,
                [(token_id, normalized_start + start, normalized_start + end) for token_id, (start, end) in zip(ids, offsets)],
            )
        ]

    def _matching_pieces(self, word: str, start: int) -> list[tuple[str, int, float]]:
        """Return vocabulary pieces that exactly match ``word[start:]``.

        Ordering intentionally matches ``pieces_by_first_character`` so the
        deterministic beam's tie handling is unchanged from the exhaustive
        implementation.
        """

        node = self.piece_tries.get(word[start])
        if node is None:
            return []
        matches: list[tuple[str, int, float]] = []
        for position in range(start, len(word)):
            child = node.get(word[position])
            if child is None:
                break
            node = child
            matches.extend(node.get("_pieces", []))
        matches.sort(key=lambda item: (-len(item[0]), -item[2], item[0]))
        return matches

    @staticmethod
    def _language_evidence(text: str, path: list[tuple[int, int, int]]) -> list[float]:
        """Candidate-specific language evidence for the later routing dot product.

        It uses only text/script and segmentation properties, never downstream
        labels.  The learned language router is the authoritative ``P(L|X)``;
        this vector says how compatible an individual segmentation is with each
        language's observed script/piece-length profile.
        """

        alpha_characters = [character for character in text if character.isalpha()]
        devanagari = sum("\u0900" <= character <= "\u097f" for character in alpha_characters)
        latin = sum("LATIN" in unicodedata.name(character, "") for character in alpha_characters)
        spanish_markers = sum(character.lower() in "áéíóúüñ" for character in text)
        english_base = max(0.05, latin - spanish_markers + 1.0)
        spanish_base = max(0.05, latin * (1.0 + 0.5 * spanish_markers) + 0.5)
        hindi_base = max(0.05, devanagari + 1.0)
        mean_piece_length = sum(end - start for _, start, end in path) / max(1, len(path))
        inverse_piece_count = 1.0 / max(1, len(path))
        evidence = [
            english_base * (1.0 + 0.03 * mean_piece_length),
            spanish_base * (1.0 + 0.05 * inverse_piece_count),
            hindi_base * (1.0 + 0.02 * mean_piece_length),
        ]
        normalizer = sum(evidence)
        return [value / normalizer for value in evidence]

    def encode_candidates(
        self,
        text: str,
        max_sequence_length: int | None = None,
        language_probabilities: list[float] | None = None,
    ) -> list[CandidateTokenization]:
        """Return filtered, temperature-normalized top-k Unigram candidates."""

        top_k = int(self.settings["max_candidates"])
        if top_k <= 0:
            raise ValueError("max_candidates must be positive.")
        normalized_text, normalized_character_spans = _normalize_with_original_spans(text)
        word_matches = list(re.finditer(r"\S+", normalized_text))
        global_paths: list[tuple[float, list[tuple[int, int, int]]]] = [(0.0, [])]
        for match in word_matches:
            word_paths = self._word_paths(match.group(), match.start(), top_k)
            if not word_paths:
                continue
            expanded = [
                (score + word_score, path + word_path)
                for score, path in global_paths
                for word_score, word_path in word_paths
            ]
            global_paths = sorted(expanded, key=lambda value: value[0], reverse=True)[:top_k]
        if not global_paths:
            return []

        candidates: list[tuple[float, float, list[int], list[tuple[int, int]], list[float]]] = []
        max_content_length = None
        if max_sequence_length is not None:
            max_content_length = max_sequence_length - self.special_token_count
        for token_score, path in global_paths:
            if max_content_length is not None:
                path = path[:max_content_length]
            content_ids = [token_id for token_id, _, _ in path]
            content_offsets = [
                _original_span(normalized_character_spans, start, end) for _, start, end in path
            ]
            input_ids = [self.bos_token_id, *content_ids, self.eos_token_id]
            special_count = len(input_ids) - len(content_ids)
            if special_count < 0:
                raise RuntimeError("Tokenizer special-token construction returned an invalid sequence.")
            # RoBERTa single sequences have a leading and trailing special token.
            offsets = [(-1, -1)] + content_offsets + [(-1, -1)]
            if len(offsets) != len(input_ids):
                offsets = [(-1, -1)] * len(input_ids)
                offsets[1 : 1 + len(content_offsets)] = content_offsets
            evidence = self._language_evidence(text, path)
            if language_probabilities is not None:
                threshold = float(self.settings.get("language_threshold", 0.0))
                retained_languages = [
                    probability if probability >= threshold else 0.0 for probability in language_probabilities
                ]
                language_total = sum(retained_languages)
                if language_total <= 0:
                    retained_languages = list(language_probabilities)
                    language_total = sum(retained_languages)
                language_probabilities = [probability / language_total for probability in retained_languages]
            language_score = 0.0 if language_probabilities is None else sum(
                probability * evidence_value for probability, evidence_value in zip(language_probabilities, evidence)
            )
            combined_score = float(self.settings["alpha"]) * token_score + float(self.settings["beta"]) * language_score
            candidates.append((combined_score, token_score, input_ids, offsets, evidence))

        temperature = float(self.settings["temperature"])
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        scores = [candidate[0] for candidate in candidates]
        maximum = max(scores)
        probabilities = [math.exp((score - maximum) / temperature) for score in scores]
        normalizer = sum(probabilities)
        probabilities = [probability / normalizer for probability in probabilities]
        minimum = float(self.settings["min_probability"])
        retained = [
            (candidate, probability)
            for candidate, probability in zip(candidates, probabilities)
            if probability >= minimum
        ]
        if not retained:
            retained = [(candidates[0], 1.0)]
        retained = retained[:top_k]
        retained_normalizer = sum(probability for _, probability in retained)
        return [
            CandidateTokenization(
                input_ids=ids,
                offsets=offsets,
                token_score=token_score,
                prior_probability=probability / retained_normalizer,
                language_evidence=evidence,
            )
            for ((_, token_score, ids, offsets, evidence), probability) in retained
        ]


def _build_unigram_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
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


def validate_probabilistic_artifact(destination: Path, expected_identity: str | None = None) -> dict[str, Any]:
    metadata_path = destination / "training_metadata.json"
    tokenizer_path = destination / "tokenizer.json"
    if not metadata_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"Incomplete probabilistic tokenizer artifact: {destination}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "probabilistic_unigram" or (
        expected_identity and metadata.get("identity_sha256") != expected_identity
    ):
        raise ValueError(f"Probabilistic artifact metadata does not match: {destination}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(destination))
    if len(tokenizer) != int(metadata["vocab_size"]):
        raise ValueError(f"Probabilistic vocabulary mismatch in {destination}")
    return metadata


def train_probabilistic_tokenizer(
    project_root: Path,
    plan_path: Path,
    tokenizer_config: dict[str, Any],
    max_sequence_length: int,
    destination: Path,
) -> dict[str, Any]:
    """Train and atomically save the Unigram candidate tokenizer artifact.

    The serialized Unigram scores are retained in tokenizer.json. Candidate ranking,
    language scoring, and character alignment consume this frozen artifact later.
    """

    special_tokens = validate_special_tokens(list(tokenizer_config["special_tokens"]))
    identity = artifact_identity(plan_path, tokenizer_config, "probabilistic_unigram")
    if prepare_artifact_destination(destination, {"tokenizer.json", "training_metadata.json"}):
        return validate_probabilistic_artifact(destination, identity)
    temporary_destination = destination.with_name(f"{destination.name}.incomplete")
    if temporary_destination.exists():
        raise FileExistsError(f"Incomplete probabilistic tokenizer artifact exists: {temporary_destination}")

    tokenizer = _build_unigram_tokenizer()
    trainer = trainers.UnigramTrainer(
        vocab_size=int(tokenizer_config["vocab_size"]),
        special_tokens=special_tokens,
        unk_token="<unk>",
        shrinking_factor=float(tokenizer_config["unigram_shrinking_factor"]),
        max_piece_length=int(tokenizer_config["unigram_max_piece_length"]),
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
        "kind": "probabilistic_unigram",
        "identity_sha256": identity,
        "vocab_size": len(fast_tokenizer),
        "plan_path": str(plan_path),
        "tokenizer_config": tokenizer_config,
        "candidate_settings": {
            key: tokenizer_config[key]
            for key in ("temperature", "max_candidates", "min_probability", "language_threshold", "alpha", "beta")
        },
    }
    write_metadata(temporary_destination / "training_metadata.json", metadata)
    temporary_destination.replace(destination)
    return validate_probabilistic_artifact(destination, identity)
