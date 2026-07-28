"""Checkpointed from-scratch MLM pretraining for the two matched groups."""

from __future__ import annotations

import json
import math
import queue
import random
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM

from src.data.pipeline import iter_prepared_corpus
from src.language.language_classifier import CulturaXLanguagePriorClassifier, LANGUAGE_TO_ID
from src.models.xlmr import (
    LanguageConditionedFusedRobertaForMaskedLM,
    build_language_conditioned_fused_xlmr_mlm,
    build_xlmr_mlm,
    model_parameter_count,
)
from src.tokenizer.probabilistic import CandidateTokenization, UnigramCandidateTokenizer
from src.tokenizer.common import build_xlmr_single_sequence
from src.training.tracking import prepare_paired_experiment, record_group_result


# Candidate construction is intentionally CPU-only.  These are initialized in
# each worker process rather than serializing the sizeable frozen tokenizer and
# scikit-learn classifier with every batch sent from the parent process.
_CANDIDATE_WORKER_TOKENIZER: UnigramCandidateTokenizer | None = None
_CANDIDATE_WORKER_LANGUAGE_PRIOR: CulturaXLanguagePriorClassifier | None = None
_CANDIDATE_WORKER_MAX_SEQUENCE_LENGTH: int | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _capture_rng_state() -> dict[str, Any]:
    """Capture every RNG whose state affects a resumable local run."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _iter_sequences(
    project_root: Path,
    plan_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    max_sequence_length: int,
    split: str,
    language: str | None = None,
    document_sequence_policy: str = "prefix_truncate",
) -> Iterator[list[int]]:
    """Yield one explicitly truncated sequence per scheduled raw document.

    This preserves the paired document schedule: both tokenizer arms receive
    the same round-robin CulturaX document at the same optimizer step.  The
    tokenizer-specific sequence is the experimental variable, not how many
    chunks a document happens to create.
    """

    if document_sequence_policy != "prefix_truncate":
        raise ValueError(f"Unsupported document_sequence_policy: {document_sequence_policy}")
    for record in iter_prepared_corpus(project_root, plan_path, split=split, balanced=True):
        if language is not None and record["language"] != language:
            continue
        special_token_count = len(build_xlmr_single_sequence(tokenizer, []))
        chunk_size = max_sequence_length - special_token_count
        if chunk_size <= 0:
            raise ValueError("max_sequence_length cannot accommodate tokenizer special tokens.")
        token_ids = tokenizer.encode(record["text"], add_special_tokens=False, truncation=True, max_length=chunk_size)
        if token_ids:
            yield build_xlmr_single_sequence(tokenizer, token_ids)


def _iter_batches(sequences: Iterator[list[int]], batch_size: int) -> Iterator[list[list[int]]]:
    batch: list[list[int]] = []
    for sequence in sequences:
        batch.append(sequence)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_training_epochs(
    batch_factory: Callable[[], Iterator[Any]], epochs: int
) -> Iterator[tuple[int, Any]]:
    """Repeat the deterministic corpus schedule for the configured epochs.

    The factory must create a fresh stream, rather than reusing an exhausted
    iterator.  Checkpoint resume remains exact because the caller stores the
    total consumed micro-batch count and skips that many items in this fixed
    epoch-major order.
    """

    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    for epoch_index in range(epochs):
        yield from ((epoch_index, batch) for batch in batch_factory())


def _mask_batch(
    sequences: list[list[int]],
    tokenizer: PreTrainedTokenizerFast,
    probability: float,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1

    generator = torch.Generator(device="cpu").manual_seed(seed)
    special_ids = torch.tensor(tokenizer.all_special_ids, dtype=torch.long)
    is_special = (input_ids.unsqueeze(-1) == special_ids).any(dim=-1)
    eligible = (attention_mask == 1) & ~is_special
    masked = (torch.rand(input_ids.shape, generator=generator) < probability) & eligible
    for row in range(masked.shape[0]):
        if not masked[row].any() and eligible[row].any():
            masked[row, torch.where(eligible[row])[0][0]] = True
    labels = input_ids.clone()
    labels[~masked] = -100

    replacement = torch.rand(input_ids.shape, generator=generator)
    mask_token_positions = masked & (replacement < 0.8)
    random_token_positions = masked & (replacement >= 0.8) & (replacement < 0.9)
    input_ids[mask_token_positions] = tokenizer.mask_token_id
    input_ids[random_token_positions] = torch.randint(
        len(tokenizer), input_ids.shape, generator=generator, dtype=torch.long
    )[random_token_positions]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def _candidate_example_from_record(
    record: dict[str, str],
    tokenizer: UnigramCandidateTokenizer,
    language_prior_classifier: CulturaXLanguagePriorClassifier,
    max_sequence_length: int,
) -> tuple[list[CandidateTokenization], int]:
    """Build one exact candidate set from one scheduled raw document.

    Keeping this transformation independent from corpus iteration makes the
    serial and multi-process paths use the same candidate selection and
    character-alignment logic.
    """

    record_language = record["language"]
    # Use one deterministic prefix per raw document, matching the BPE control
    # path's document-level schedule.  Candidate spans retain their
    # original-document coordinates for fusion and auditability.
    encoded = tokenizer.tokenizer(
        record["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_sequence_length - len(build_xlmr_single_sequence(tokenizer.tokenizer, [])),
    )
    offsets = list(encoded["offset_mapping"])
    content_capacity = max_sequence_length - len(build_xlmr_single_sequence(tokenizer.tokenizer, []))
    if content_capacity <= 0:
        raise ValueError("max_sequence_length cannot accommodate tokenizer special tokens.")
    chunk_offsets = offsets[:content_capacity]
    if not chunk_offsets:
        raise RuntimeError(
            "The proposed tokenizer produced no content offsets for a non-empty scheduled document; "
            "stopping prevents the paired raw-document schedule from silently diverging from BPE."
        )
    character_start, character_end = chunk_offsets[0][0], chunk_offsets[-1][1]
    if character_end <= character_start:
        raise RuntimeError(
            "The proposed tokenizer produced an invalid content span for a scheduled document; "
            "stopping prevents an unmatched BPE/proposed training step."
        )
    chunk_text = record["text"][character_start:character_end]
    language_probabilities = language_prior_classifier.predict_probabilities(chunk_text)
    candidates = tokenizer.encode_candidates(
        chunk_text,
        max_sequence_length=max_sequence_length,
        language_probabilities=language_probabilities,
    )
    if not candidates:
        raise RuntimeError(
            "No probabilistic tokenization candidate was available for a scheduled document; "
            "stopping prevents the paired raw-document schedule from silently diverging from BPE."
        )
    # The model's within-document overlap calculations work on either origin,
    # but retaining original offsets makes saved diagnostics auditable and
    # satisfies the character-alignment contract.
    return [
        CandidateTokenization(
            input_ids=candidate.input_ids,
            offsets=[
                (span_start + character_start, span_end + character_start)
                if span_start >= 0
                else (span_start, span_end)
                for span_start, span_end in candidate.offsets
            ],
            token_score=candidate.token_score,
            prior_probability=candidate.prior_probability,
            language_evidence=candidate.language_evidence,
        )
        for candidate in candidates
    ], LANGUAGE_TO_ID[record_language]


def _iter_candidate_examples(
    project_root: Path,
    plan_path: Path,
    tokenizer: UnigramCandidateTokenizer,
    language_prior_classifier: CulturaXLanguagePriorClassifier,
    max_sequence_length: int,
    split: str,
    language: str | None = None,
    document_sequence_policy: str = "prefix_truncate",
) -> Iterator[tuple[list[CandidateTokenization], int]]:
    """Build top-k candidate paths from the frozen Unigram artifact on demand."""

    if document_sequence_policy != "prefix_truncate":
        raise ValueError(f"Unsupported document_sequence_policy: {document_sequence_policy}")
    for record in iter_prepared_corpus(project_root, plan_path, split=split, balanced=True):
        if language is not None and record["language"] != language:
            continue
        yield _candidate_example_from_record(
            {"text": str(record["text"]), "language": str(record["language"])},
            tokenizer,
            language_prior_classifier,
            max_sequence_length,
        )


def _iter_candidate_batches(
    examples: Iterator[tuple[list[CandidateTokenization], int]],
    batch_size: int,
) -> Iterator[list[tuple[list[CandidateTokenization], int]]]:
    batch: list[tuple[list[CandidateTokenization], int]] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_candidate_record_batches(
    project_root: Path,
    plan_path: Path,
    split: str,
    batch_size: int,
    document_sequence_policy: str,
) -> Iterator[list[dict[str, str]]]:
    """Yield fixed-size, ordered raw-document batches for CPU workers."""

    if document_sequence_policy != "prefix_truncate":
        raise ValueError(f"Unsupported document_sequence_policy: {document_sequence_policy}")
    batch: list[dict[str, str]] = []
    for record in iter_prepared_corpus(project_root, plan_path, split=split, balanced=True):
        batch.append({"text": str(record["text"]), "language": str(record["language"])})
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _initialize_candidate_worker(
    tokenizer_directory: str,
    language_classifier_directory: str,
    probabilistic_settings: dict[str, Any],
    max_sequence_length: int,
) -> None:
    """Load immutable candidate assets once per spawned CPU worker."""

    global _CANDIDATE_WORKER_TOKENIZER
    global _CANDIDATE_WORKER_LANGUAGE_PRIOR
    global _CANDIDATE_WORKER_MAX_SEQUENCE_LENGTH
    tokenizer = UnigramCandidateTokenizer.from_pretrained(tokenizer_directory)
    tokenizer.configure_candidate_selection(probabilistic_settings)
    _CANDIDATE_WORKER_TOKENIZER = tokenizer
    _CANDIDATE_WORKER_LANGUAGE_PRIOR = CulturaXLanguagePriorClassifier.load(Path(language_classifier_directory))
    _CANDIDATE_WORKER_MAX_SEQUENCE_LENGTH = int(max_sequence_length)


def _build_candidate_record_batch_in_worker(
    records: list[dict[str, str]],
) -> list[tuple[list[CandidateTokenization], int]]:
    """Construct one ordered candidate batch in a CPU worker process."""

    if (
        _CANDIDATE_WORKER_TOKENIZER is None
        or _CANDIDATE_WORKER_LANGUAGE_PRIOR is None
        or _CANDIDATE_WORKER_MAX_SEQUENCE_LENGTH is None
    ):
        raise RuntimeError("Candidate worker assets were not initialized.")
    return [
        _candidate_example_from_record(
            record,
            _CANDIDATE_WORKER_TOKENIZER,
            _CANDIDATE_WORKER_LANGUAGE_PRIOR,
            _CANDIDATE_WORKER_MAX_SEQUENCE_LENGTH,
        )
        for record in records
    ]


def _iter_parallel_candidate_batches(
    project_root: Path,
    plan_path: Path,
    tokenizer_directory: Path,
    language_classifier_directory: Path,
    probabilistic_settings: dict[str, Any],
    max_sequence_length: int,
    split: str,
    batch_size: int,
    document_sequence_policy: str,
    worker_count: int,
    buffer_batches: int,
) -> Iterator[list[tuple[list[CandidateTokenization], int]]]:
    """Prepare exact candidate batches concurrently without changing order.

    The parent remains the sole corpus reader.  Workers receive only bounded
    batches of already-scheduled raw records, return their same-index candidate
    batch, and are consumed in submission order.  The settings and frozen
    artifacts are loaded once by each process, not re-trained or modified.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive for parallel candidate preparation.")
    if buffer_batches <= 0:
        raise ValueError("buffer_batches must be positive for parallel candidate preparation.")

    record_batches = iter(
        _iter_candidate_record_batches(
            project_root,
            plan_path,
            split,
            batch_size,
            document_sequence_policy,
        )
    )
    maximum_in_flight = worker_count * buffer_batches
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_candidate_worker,
        initargs=(
            str(tokenizer_directory),
            str(language_classifier_directory),
            dict(probabilistic_settings),
            int(max_sequence_length),
        ),
    ) as executor:
        pending: deque[Any] = deque()

        def submit_next() -> bool:
            try:
                records = next(record_batches)
            except StopIteration:
                return False
            pending.append(executor.submit(_build_candidate_record_batch_in_worker, records))
            return True

        for _ in range(maximum_in_flight):
            if not submit_next():
                break
        while pending:
            # Retrieving futures in submission order makes the process pool an
            # implementation detail: candidate-to-document assignment remains
            # exactly the serial round-robin corpus schedule.
            yield pending.popleft().result()
            submit_next()


def _prefetch_iterator(source: Iterator[Any], buffer_size: int) -> Iterator[Any]:
    """Overlap deterministic CPU candidate construction with GPU training.

    The producer is deliberately single-threaded, so it preserves corpus order
    and all candidate contents exactly.  It only keeps a bounded number of
    already-computed batches in memory while the consumer runs the previous
    GPU forward/backward pass.
    """

    if buffer_size <= 0:
        yield from source
        return
    items: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=buffer_size)
    stop = threading.Event()

    def publish(kind: str, value: Any) -> bool:
        while not stop.is_set():
            try:
                items.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in source:
                if not publish("item", item):
                    return
        except BaseException as error:  # propagate worker failure in training order
            publish("error", error)
        finally:
            publish("done", None)

    producer = threading.Thread(target=produce, name="candidate-batch-prefetch", daemon=True)
    producer.start()
    try:
        while True:
            kind, value = items.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        stop.set()
        producer.join(timeout=5.0)


def _build_sparse_alignment_edges(
    char_spans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact character-overlap edges for sparse candidate projection.

    Candidate and reference token spans are monotonic segmentations of the
    same source text.  There are therefore only O(tokens) true overlap edges,
    rather than O(tokens squared).  Packing those edges on CPU keeps the model
    path mathematically identical while avoiding a dense 5 x 512 x 512
    alignment contraction in every fused forward pass.
    """

    if char_spans.ndim != 4 or char_spans.shape[-1] != 2:
        raise ValueError("char_spans must have shape [batch, candidates, tokens, 2].")
    batch_size, candidate_count, _, _ = char_spans.shape
    candidate_start = char_spans[..., 0]
    candidate_end = char_spans[..., 1]
    reference_start = char_spans[:, 0, :, 0]
    reference_end = char_spans[:, 0, :, 1]
    valid_candidate = (candidate_start >= 0) & (candidate_end > candidate_start)
    valid_reference = (reference_start >= 0) & (reference_end > reference_start)
    overlap_lengths = (
        torch.minimum(candidate_end[:, :, :, None], reference_end[:, None, None, :])
        - torch.maximum(candidate_start[:, :, :, None], reference_start[:, None, None, :])
    ).clamp_min(0)
    overlap_mask = (
        (overlap_lengths > 0)
        & valid_candidate[:, :, :, None]
        & valid_reference[:, None, None, :]
    )
    edge_counts = overlap_mask.sum(dim=(-1, -2))
    maximum_edges = int(edge_counts.max().item())
    reference_indices = torch.zeros((batch_size, candidate_count, maximum_edges), dtype=torch.long)
    candidate_indices = torch.zeros_like(reference_indices)
    overlap_weights = torch.zeros((batch_size, candidate_count, maximum_edges), dtype=torch.float)
    for batch_index in range(batch_size):
        for candidate_index in range(candidate_count):
            candidate_positions, reference_positions = torch.where(overlap_mask[batch_index, candidate_index])
            edge_count = len(candidate_positions)
            if edge_count == 0:
                continue
            reference_indices[batch_index, candidate_index, :edge_count] = reference_positions
            candidate_indices[batch_index, candidate_index, :edge_count] = candidate_positions
            overlap_weights[batch_index, candidate_index, :edge_count] = overlap_lengths[
                batch_index, candidate_index, candidate_positions, reference_positions
            ].to(torch.float)
    return reference_indices, candidate_indices, overlap_weights


def _mask_candidate_batch(
    examples: list[tuple[list[CandidateTokenization], int]],
    tokenizer: UnigramCandidateTokenizer,
    probability: float,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pad and dynamically mask top-k paths using reference-character spans.

    Labels are the first (highest-scoring) path.  The same original text spans
    are masked in every alternate path, so fusion cannot reveal a reference
    token through an unmasked differently segmented candidate.
    """

    fast_tokenizer = tokenizer.tokenizer
    batch_size = len(examples)
    max_candidates = max(len(candidates) for candidates, _ in examples)
    max_length = max(len(candidate.input_ids) for candidates, _ in examples for candidate in candidates)
    input_ids = torch.full(
        (batch_size, max_candidates, max_length), fast_tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    char_spans = torch.full((batch_size, max_candidates, max_length, 2), -1, dtype=torch.long)
    prior_probabilities = torch.zeros((batch_size, max_candidates), dtype=torch.float)
    language_evidence = torch.zeros((batch_size, max_candidates, 3), dtype=torch.float)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    language_labels = torch.empty(batch_size, dtype=torch.long)
    for row, (candidates, language_label) in enumerate(examples):
        language_labels[row] = language_label
        for candidate_index, candidate in enumerate(candidates):
            length = len(candidate.input_ids)
            input_ids[row, candidate_index, :length] = torch.tensor(candidate.input_ids, dtype=torch.long)
            attention_mask[row, candidate_index, :length] = 1
            char_spans[row, candidate_index, :length] = torch.tensor(candidate.offsets, dtype=torch.long)
            prior_probabilities[row, candidate_index] = candidate.prior_probability
            language_evidence[row, candidate_index] = torch.tensor(candidate.language_evidence, dtype=torch.float)
            candidate_mask[row, candidate_index] = True

    reference_ids = input_ids[:, 0].clone()
    reference_attention = attention_mask[:, 0]
    special_ids = torch.tensor(fast_tokenizer.all_special_ids, dtype=torch.long)
    is_special = (reference_ids.unsqueeze(-1) == special_ids).any(dim=-1)
    eligible = (reference_attention == 1) & ~is_special
    generator = torch.Generator(device="cpu").manual_seed(seed)
    masked_reference = (torch.rand(reference_ids.shape, generator=generator) < probability) & eligible
    for row in range(batch_size):
        if not masked_reference[row].any() and eligible[row].any():
            masked_reference[row, torch.where(eligible[row])[0][0]] = True
    labels = reference_ids.clone()
    labels[~masked_reference] = -100
    replacements = torch.rand(reference_ids.shape, generator=generator)
    # Align mask decisions by original character spans in one batched operation.
    # This is equivalent to the former nested Python loop: each candidate token
    # adopts the replacement decision of its first overlapping masked reference
    # token.  It preserves the no-leakage MLM rule while avoiding tens of
    # thousands of scalar tensor operations per microbatch.
    candidate_start = char_spans[..., 0]
    candidate_end = char_spans[..., 1]
    reference_spans = char_spans[:, 0]
    reference_start = reference_spans[..., 0]
    reference_end = reference_spans[..., 1]
    valid_candidate = (candidate_start >= 0) & (candidate_end > candidate_start)
    valid_reference = (reference_start >= 0) & (reference_end > reference_start)
    overlaps = (
        (candidate_start[:, :, :, None] < reference_end[:, None, None, :])
        & (candidate_end[:, :, :, None] > reference_start[:, None, None, :])
        & valid_candidate[:, :, :, None]
        & valid_reference[:, None, None, :]
        & masked_reference[:, None, None, :]
    )
    has_match = overlaps.any(dim=-1)
    first_match = overlaps.to(torch.int64).argmax(dim=-1)
    replacement_for_candidate = replacements[:, None, :].expand(-1, max_candidates, -1).gather(
        2, first_match
    )
    mask_positions = has_match & (replacement_for_candidate < 0.8)
    random_positions = has_match & (replacement_for_candidate >= 0.8) & (replacement_for_candidate < 0.9)
    input_ids.masked_fill_(mask_positions, fast_tokenizer.mask_token_id)
    random_ids = torch.randint(len(fast_tokenizer), input_ids.shape, generator=generator, dtype=torch.long)
    input_ids[random_positions] = random_ids[random_positions]
    alignment_reference_indices, alignment_candidate_indices, alignment_overlap_weights = _build_sparse_alignment_edges(
        char_spans
    )
    return {
        "candidate_input_ids": input_ids.to(device),
        "candidate_attention_mask": attention_mask.to(device),
        "candidate_char_spans": char_spans.to(device),
        "candidate_prior_probabilities": prior_probabilities.to(device),
        "candidate_language_evidence": language_evidence.to(device),
        "candidate_mask": candidate_mask.to(device),
        "candidate_alignment_reference_indices": alignment_reference_indices.to(device),
        "candidate_alignment_candidate_indices": alignment_candidate_indices.to(device),
        "candidate_alignment_overlap_weights": alignment_overlap_weights.to(device),
        "labels": labels.to(device),
        "language_labels": language_labels.to(device),
    }


def _linear_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, schedule)


def _evaluate(
    model: RobertaForMaskedLM,
    project_root: Path,
    plan_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    max_sequence_length: int,
    training: dict[str, Any],
    language: str,
    device: torch.device,
    seed: int,
    split: str = "validation",
) -> float:
    model.eval()
    losses: list[float] = []
    batches = _iter_batches(
        _iter_sequences(
            project_root,
            plan_path,
            tokenizer,
            max_sequence_length,
            split,
            language,
            str(training["document_sequence_policy"]),
        ),
        int(training["batch_size"]),
    )
    with torch.no_grad():
        for index, batch in enumerate(islice(batches, int(training["validation_batches_per_language"]))):
            inputs = _mask_batch(batch, tokenizer, float(training["mask_probability"]), seed + index, device)
            losses.append(float(model(**inputs).loss.detach().cpu()))
    if not losses:
        raise ValueError(f"No validation batches available for {language}")
    return sum(losses) / len(losses)


def _save_checkpoint(
    checkpoint_directory: Path,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    optimizer: AdamW,
    scheduler: LambdaLR,
    state: dict[str, Any],
) -> None:
    temporary_directory = checkpoint_directory.with_name(f"{checkpoint_directory.name}.incomplete")
    if temporary_directory.exists():
        raise FileExistsError(
            f"Incomplete checkpoint exists and will not be overwritten: {temporary_directory}. "
            "Inspect, archive, or explicitly remove it before retrying."
        )
    backup_directory = checkpoint_directory.with_name(f"{checkpoint_directory.name}.previous")
    if backup_directory.exists():
        if checkpoint_directory.exists():
            raise FileExistsError(
                f"Ambiguous checkpoint recovery state: {checkpoint_directory} and {backup_directory}."
            )
        print(f"Restoring checkpoint from: {backup_directory}")
        backup_directory.replace(checkpoint_directory)
    temporary_directory.mkdir(parents=True, exist_ok=False)
    print(f"Saving checkpoint to: {temporary_directory}")
    model.save_pretrained(str(temporary_directory))
    tokenizer.save_pretrained(str(temporary_directory / "tokenizer"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
            "rng_state": _capture_rng_state(),
        },
        temporary_directory / "trainer_state.pt",
    )
    (temporary_directory / "metrics.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if checkpoint_directory.exists():
        checkpoint_directory.replace(backup_directory)
    try:
        temporary_directory.replace(checkpoint_directory)
    except Exception:
        if backup_directory.exists() and not checkpoint_directory.exists():
            backup_directory.replace(checkpoint_directory)
        raise
    if backup_directory.exists():
        shutil.rmtree(backup_directory)


def _resolve_checkpoint(destination: Path, preference: str = "latest") -> Path:
    """Locate a complete model checkpoint for resume or evaluation."""

    candidates = [path for path in destination.glob("checkpoint-*") if (path / "trainer_state.pt").is_file()]
    candidates.sort(key=lambda path: path.name)
    if preference == "latest" and candidates:
        return candidates[-1]
    preferred_names = ("best", "final") if preference == "best" else ("final", "best")
    for name in preferred_names:
        named = destination / name
        if named.is_dir() and (named / "trainer_state.pt").is_file():
            return named
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No complete checkpoint is available below: {destination}")


def _restore_optimizer_state(
    optimizer: AdamW,
    scheduler: LambdaLR,
    checkpoint_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Restore optimizer/scheduler state and move state tensors to ``device``."""

    state_path = checkpoint_directory / "trainer_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload.get("rng_state"))
    return dict(payload.get("state", {}))


def _resume_after_microbatches(
    batches: Iterator[Any],
    consumed_microbatches: int,
) -> Iterator[Any]:
    """Advance a deterministic corpus iterator to the saved batch boundary."""

    for _ in range(consumed_microbatches):
        try:
            next(batches)
        except StopIteration as error:
            raise RuntimeError(
                "Checkpoint data cursor is beyond the current corpus. "
                "Refuse to repeat data under a changed corpus/configuration."
            ) from error
    return batches


def _prune_periodic_checkpoints(destination: Path, maximum_retained: int) -> None:
    """Retain only the configured number of numbered resumable checkpoints."""

    if maximum_retained <= 0:
        return
    checkpoints = sorted(
        (path for path in destination.glob("checkpoint-*") if (path / "trainer_state.pt").is_file()),
        key=lambda path: path.name,
    )
    for stale in checkpoints[:-maximum_retained]:
        print(f"Removing superseded checkpoint: {stale}")
        shutil.rmtree(stale)


def _save_json_result(destination: Path, payload: dict[str, Any]) -> None:
    """Atomically persist an evaluation result in the configured results tree."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    print(f"Saving evaluation result to: {destination}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _evaluate_fused(
    model: LanguageConditionedFusedRobertaForMaskedLM,
    project_root: Path,
    plan_path: Path,
    tokenizer: UnigramCandidateTokenizer,
    language_prior_classifier: CulturaXLanguagePriorClassifier,
    model_settings: dict[str, Any],
    training: dict[str, Any],
    language: str,
    device: torch.device,
    seed: int,
    split: str = "validation",
) -> dict[str, float]:
    """Evaluate the actual fused forward path, including its routing behavior."""

    model.eval()
    mlm_losses: list[float] = []
    routing_losses: list[float] = []
    candidate_weight_means: list[float] = []
    batches = _iter_candidate_batches(
        _iter_candidate_examples(
            project_root,
            plan_path,
            tokenizer,
            language_prior_classifier,
            int(model_settings["max_sequence_length"]),
            split,
            language,
            str(training["document_sequence_policy"]),
        ),
        int(training["batch_size"]),
    )
    with torch.no_grad():
        for index, batch in enumerate(islice(batches, int(training["validation_batches_per_language"]))):
            inputs = _mask_candidate_batch(batch, tokenizer, float(training["mask_probability"]), seed + index, device)
            output = model(**inputs)
            if output.mlm_loss is None:
                raise RuntimeError("Fused evaluation did not produce an MLM loss.")
            mlm_losses.append(float(output.mlm_loss.detach().cpu()))
            if output.routing_loss is not None:
                routing_losses.append(float(output.routing_loss.detach().cpu()))
            candidate_weight_means.append(float(output.candidate_weights[:, 0].mean().detach().cpu()))
    if not mlm_losses:
        raise ValueError(f"No validation batches available for {language}")
    return {
        "mlm_loss": sum(mlm_losses) / len(mlm_losses),
        "routing_loss": sum(routing_losses) / len(routing_losses) if routing_losses else 0.0,
        "reference_candidate_weight": sum(candidate_weight_means) / len(candidate_weight_means),
    }


def train_language_conditioned_fused_mlm(
    project_root: Path,
    plan_path: Path,
    tokenizer_directory: Path,
    language_classifier_directory: Path,
    model_settings: dict[str, Any],
    probabilistic_settings: dict[str, Any],
    training: dict[str, Any],
    destination: Path,
    runtime_mode: str,
    logs_directory: Path,
    results_directory: Path,
) -> dict[str, Any]:
    """Train the proposed top-k candidate-fusion MLM from scratch.

    Unlike the BPE baseline trainer, every optimizer step calls the fused
    model's candidate-based forward method.  The classifier probabilities,
    candidate weights, and auxiliary routing loss are all TensorBoard-logged.
    """

    if runtime_mode not in {"train", "resume", "evaluate"}:
        raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
    tokenizer = UnigramCandidateTokenizer.from_pretrained(tokenizer_directory)
    candidate_selection_settings = tokenizer.configure_candidate_selection(probabilistic_settings)
    print(f"probabilistic runtime candidate selection: {candidate_selection_settings}")
    language_prior_classifier = CulturaXLanguagePriorClassifier.load(language_classifier_directory)
    candidate_worker_count = int(training.get("candidate_preparation_workers", 0))
    candidate_buffer_batches = int(training.get("candidate_preparation_buffer_batches", 1))
    if candidate_worker_count < 0:
        raise ValueError("training.candidate_preparation_workers must be non-negative.")
    if candidate_worker_count and candidate_buffer_batches <= 0:
        raise ValueError("training.candidate_preparation_buffer_batches must be positive.")
    candidate_preparation_settings = {
        "workers": candidate_worker_count,
        "buffer_batches": candidate_buffer_batches,
        "serial_prefetch_batches": int(training.get("candidate_prefetch_batches", 0)),
    }
    print(f"probabilistic candidate preparation: {candidate_preparation_settings}")
    experiment_directory = prepare_paired_experiment(
        project_root,
        plan_path,
        "probabilistic_model",
        tokenizer_directory,
        model_settings,
        training,
        group_settings={
            "candidate_selection": candidate_selection_settings,
            "candidate_preparation": candidate_preparation_settings,
        },
    )
    set_global_seed(int(training["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_directory: Path | None = None
    if runtime_mode == "train":
        model = build_language_conditioned_fused_xlmr_mlm(
            model_settings, len(tokenizer.tokenizer), probabilistic_settings
        ).to(device)
    else:
        checkpoint_directory = _resolve_checkpoint(destination, "latest" if runtime_mode == "resume" else "best")
        model = LanguageConditionedFusedRobertaForMaskedLM.from_pretrained(checkpoint_directory).to(device)
    print(f"probabilistic_fused_model parameters: {model_parameter_count(model):,}")
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    total_steps = int(training["maximum_optimizer_steps"])
    scheduler = _linear_scheduler(optimizer, int(total_steps * float(training["warmup_ratio"])), total_steps)
    use_fp16 = device.type == "cuda" and training["precision"] == "fp16_or_bfloat16_if_available"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    restored_state = (
        _restore_optimizer_state(optimizer, scheduler, checkpoint_directory, device)
        if runtime_mode == "resume" and checkpoint_directory is not None
        else {}
    )
    if runtime_mode == "evaluate":
        test_metrics = {
            language: _evaluate_fused(
                model,
                project_root,
                plan_path,
                tokenizer,
                language_prior_classifier,
                model_settings,
                training,
                language,
                device,
                int(training["seed"]),
                split="test",
            )
            for language in ("en", "es", "hi")
        }
        result = {
            "runtime_mode": "evaluate",
            "checkpoint": str(checkpoint_directory),
            "test_per_language": test_metrics,
            "test_macro_mlm_loss": sum(value["mlm_loss"] for value in test_metrics.values()) / len(test_metrics),
        }
        _save_json_result(results_directory / "mlm" / "probabilistic_test_metrics.json", result)
        record_group_result(experiment_directory, "probabilistic_model", result)
        return result

    destination.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logs_directory / "probabilistic_fused_model"))
    best_loss = float(restored_state.get("best_macro_validation_loss", math.inf))
    no_improvement = int(restored_state.get("no_improvement", 0))
    global_step = int(restored_state.get("global_step", 0))
    optimizer.zero_grad(set_to_none=True)
    accumulation_steps = int(training["gradient_accumulation_steps"])
    def fused_batch_factory() -> Iterator[list[tuple[list[CandidateTokenization], int]]]:
        if candidate_worker_count:
            return _iter_parallel_candidate_batches(
                project_root,
                plan_path,
                tokenizer_directory,
                language_classifier_directory,
                probabilistic_settings,
                int(model_settings["max_sequence_length"]),
                "train",
                int(training["batch_size"]),
                str(training["document_sequence_policy"]),
                candidate_worker_count,
                candidate_buffer_batches,
            )
        return _iter_candidate_batches(
            _iter_candidate_examples(
                project_root,
                plan_path,
                tokenizer,
                language_prior_classifier,
                int(model_settings["max_sequence_length"]),
                "train",
                document_sequence_policy=str(training["document_sequence_policy"]),
            ),
            int(training["batch_size"]),
        )

    train_batches = _iter_training_epochs(fused_batch_factory, int(training["epochs"]))
    consumed_microbatches = int(restored_state.get("consumed_microbatches", global_step * accumulation_steps))
    if runtime_mode == "resume":
        train_batches = _resume_after_microbatches(train_batches, consumed_microbatches)
    if not candidate_worker_count:
        train_batches = _prefetch_iterator(
            train_batches, int(training.get("candidate_prefetch_batches", 0))
        )
    microbatches_seen = consumed_microbatches
    pending_microbatches = 0
    running_mlm_loss = 0.0
    running_total_loss = 0.0
    running_masked_tokens = 0
    running_masked_correct = 0
    running_processed_tokens = 0
    running_sequences = 0
    processed_tokens = int(restored_state.get("processed_tokens", 0))
    processed_sequences = int(restored_state.get("processed_sequences", 0))
    started_at = time.perf_counter()
    stop_training = False
    for epoch_index, batch in train_batches:
        if global_step >= total_steps:
            break
        model.train()
        inputs = _mask_candidate_batch(batch, tokenizer, float(training["mask_probability"]), int(training["seed"]) + global_step, device)
        autocast = torch.cuda.amp.autocast if use_fp16 else nullcontext
        with autocast():
            output = model(**inputs)
            if output.loss is None or output.mlm_loss is None:
                raise RuntimeError("Fused MLM training requires both MLM labels and a total loss.")
            loss = output.loss / accumulation_steps
        masked_positions = inputs["labels"] != -100
        running_masked_tokens += int(masked_positions.sum().item())
        running_masked_correct += int(
            (output.logits.argmax(dim=-1)[masked_positions] == inputs["labels"][masked_positions]).sum().item()
        )
        running_processed_tokens += int(inputs["candidate_attention_mask"][:, 0].sum().item())
        running_sequences += len(batch)
        scaler.scale(loss).backward() if use_fp16 else loss.backward()
        pending_microbatches += 1
        microbatches_seen += 1
        running_mlm_loss += float(output.mlm_loss.detach().cpu())
        running_total_loss += float(output.loss.detach().cpu())
        if pending_microbatches < accumulation_steps:
            continue
        if use_fp16:
            scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clipping"])).detach().cpu())
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        writer.add_scalar("train/mlm_loss", running_mlm_loss / pending_microbatches, global_step)
        writer.add_scalar("train/total_loss", running_total_loss / pending_microbatches, global_step)
        writer.add_scalar("train/routing_auxiliary_loss", float(output.routing_loss.detach().cpu()), global_step)
        writer.add_scalar("train/gradient_norm", gradient_norm, global_step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("train/epoch", epoch_index + 1, global_step)
        processed_tokens += running_processed_tokens
        processed_sequences += running_sequences
        writer.add_scalar("train/masked_token_accuracy", running_masked_correct / max(1, running_masked_tokens), global_step)
        writer.add_scalar("train/processed_tokens", processed_tokens, global_step)
        writer.add_scalar("train/processed_sequences", processed_sequences, global_step)
        writer.add_scalar("train/elapsed_seconds", time.perf_counter() - started_at, global_step)
        if device.type == "cuda":
            writer.add_scalar("system/gpu_max_memory_bytes", torch.cuda.max_memory_allocated(device), global_step)
        routing_entropy = -(output.routing_probabilities * output.routing_probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()
        writer.add_scalar("routing/entropy", float(routing_entropy.detach().cpu()), global_step)
        for language, language_index in LANGUAGE_TO_ID.items():
            writer.add_scalar(
                f"routing/probability_{language}",
                float(output.routing_probabilities[:, language_index].mean().detach().cpu()),
                global_step,
            )
        for candidate_index in range(output.candidate_weights.shape[1]):
            writer.add_scalar(
                f"candidate_weights/candidate_{candidate_index}",
                float(output.candidate_weights[:, candidate_index].mean().detach().cpu()),
                global_step,
            )
        pending_microbatches = 0
        running_mlm_loss = 0.0
        running_total_loss = 0.0
        running_masked_tokens = 0
        running_masked_correct = 0
        running_processed_tokens = 0
        running_sequences = 0

        if global_step % int(training["evaluation_interval"]) == 0:
            per_language = {
                language: _evaluate_fused(
                    model,
                    project_root,
                    plan_path,
                    tokenizer,
                    language_prior_classifier,
                    model_settings,
                    training,
                    language,
                    device,
                    global_step,
                )
                for language in ("en", "es", "hi")
            }
            validation_loss = sum(metrics["mlm_loss"] for metrics in per_language.values()) / len(per_language)
            writer.add_scalar("validation/macro_mlm_loss", validation_loss, global_step)
            for language, metrics in per_language.items():
                writer.add_scalar(f"validation/{language}_mlm_loss", metrics["mlm_loss"], global_step)
                writer.add_scalar(f"validation/{language}_routing_loss", metrics["routing_loss"], global_step)
                writer.add_scalar(
                    f"validation/{language}_reference_candidate_weight",
                    metrics["reference_candidate_weight"],
                    global_step,
                )
            state = {
                "global_step": global_step,
                "macro_validation_loss": validation_loss,
                "best_macro_validation_loss": best_loss,
                "no_improvement": no_improvement,
                "consumed_microbatches": microbatches_seen,
                "processed_tokens": processed_tokens,
                "processed_sequences": processed_sequences,
                "per_language": per_language,
                "epoch": epoch_index + 1,
            }
            if validation_loss < best_loss:
                best_loss = validation_loss
                no_improvement = 0
                state["best_macro_validation_loss"] = best_loss
                state["no_improvement"] = no_improvement
                _save_checkpoint(destination / "best", model, tokenizer.tokenizer, optimizer, scheduler, state)
            else:
                no_improvement += 1
            if no_improvement >= int(training["early_stopping_patience"]):
                stop_training = True
        if global_step % int(training["checkpoint_interval"]) == 0:
            _save_checkpoint(
                destination / f"checkpoint-{global_step:08d}",
                model,
                tokenizer.tokenizer,
                optimizer,
                scheduler,
                {
                    "global_step": global_step,
                    "best_macro_validation_loss": best_loss,
                    "no_improvement": no_improvement,
                    "consumed_microbatches": microbatches_seen,
                    "processed_tokens": processed_tokens,
                    "processed_sequences": processed_sequences,
                    "epoch": epoch_index + 1,
                },
            )
            _prune_periodic_checkpoints(destination, int(training["maximum_retained_checkpoints"]))
        if stop_training:
            break

    if pending_microbatches:
        raise RuntimeError("Corpus ended mid-gradient-accumulation; rerun with a batch size or accumulation factor that yields full optimizer steps.")
    final_state = {
        "global_step": global_step,
        "best_macro_validation_loss": best_loss,
        "no_improvement": no_improvement,
        "consumed_microbatches": microbatches_seen,
        "processed_tokens": processed_tokens,
        "processed_sequences": processed_sequences,
        "model_kind": "language_conditioned_top_k_fusion",
        "epoch": epoch_index + 1 if "epoch_index" in locals() else 0,
    }
    _save_checkpoint(destination / "final", model, tokenizer.tokenizer, optimizer, scheduler, final_state)
    writer.close()
    selected_checkpoint = _resolve_checkpoint(destination, "best")
    selected_model = LanguageConditionedFusedRobertaForMaskedLM.from_pretrained(selected_checkpoint).to(device)
    test_metrics = {
        language: _evaluate_fused(
            selected_model,
            project_root,
            plan_path,
            tokenizer,
            language_prior_classifier,
            model_settings,
            training,
            language,
            device,
            int(training["seed"]),
            split="test",
        )
        for language in ("en", "es", "hi")
    }
    final_state.update(
        {
            "selected_checkpoint": str(selected_checkpoint),
            "test_per_language": test_metrics,
            "test_macro_mlm_loss": sum(value["mlm_loss"] for value in test_metrics.values()) / len(test_metrics),
        }
    )
    _save_json_result(results_directory / "mlm" / "probabilistic_test_metrics.json", final_state)
    record_group_result(experiment_directory, "probabilistic_model", final_state)
    return final_state


def train_from_scratch_mlm(
    project_root: Path,
    plan_path: Path,
    tokenizer_directory: Path,
    model_settings: dict[str, Any],
    training: dict[str, Any],
    destination: Path,
    group_name: str,
    runtime_mode: str,
    logs_directory: Path,
    results_directory: Path,
) -> dict[str, Any]:
    """Train one matched MLM group and save best/final resumable checkpoints."""

    if runtime_mode not in {"train", "resume", "evaluate"}:
        raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_directory))
    experiment_directory = prepare_paired_experiment(
        project_root,
        plan_path,
        group_name,
        tokenizer_directory,
        model_settings,
        training,
    )
    set_global_seed(int(training["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_directory: Path | None = None
    if runtime_mode == "train":
        model = build_xlmr_mlm(model_settings, len(tokenizer)).to(device)
    else:
        checkpoint_directory = _resolve_checkpoint(destination, "latest" if runtime_mode == "resume" else "best")
        model = RobertaForMaskedLM.from_pretrained(str(checkpoint_directory)).to(device)
    print(f"{group_name} parameters: {model_parameter_count(model):,}")

    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    total_steps = int(training["maximum_optimizer_steps"])
    scheduler = _linear_scheduler(optimizer, int(total_steps * float(training["warmup_ratio"])), total_steps)
    use_fp16 = device.type == "cuda" and training["precision"] == "fp16_or_bfloat16_if_available"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    restored_state = (
        _restore_optimizer_state(optimizer, scheduler, checkpoint_directory, device)
        if runtime_mode == "resume" and checkpoint_directory is not None
        else {}
    )
    if runtime_mode == "evaluate":
        test_metrics = {
            language: _evaluate(
                model,
                project_root,
                plan_path,
                tokenizer,
                int(model_settings["max_sequence_length"]),
                training,
                language,
                device,
                int(training["seed"]),
                split="test",
            )
            for language in ("en", "es", "hi")
        }
        result = {
            "runtime_mode": "evaluate",
            "checkpoint": str(checkpoint_directory),
            "test_per_language": test_metrics,
            "test_macro_mlm_loss": sum(test_metrics.values()) / len(test_metrics),
        }
        _save_json_result(results_directory / "mlm" / f"{group_name}_test_metrics.json", result)
        record_group_result(experiment_directory, group_name, result)
        return result
    destination.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logs_directory / group_name))

    best_loss = float(restored_state.get("best_macro_validation_loss", math.inf))
    no_improvement = int(restored_state.get("no_improvement", 0))
    global_step = int(restored_state.get("global_step", 0))
    optimizer.zero_grad(set_to_none=True)
    accumulation_steps = int(training["gradient_accumulation_steps"])
    def baseline_batch_factory() -> Iterator[list[list[int]]]:
        return _iter_batches(
            _iter_sequences(
                project_root,
                plan_path,
                tokenizer,
                int(model_settings["max_sequence_length"]),
                "train",
                document_sequence_policy=str(training["document_sequence_policy"]),
            ),
            int(training["batch_size"]),
        )

    train_batches = _iter_training_epochs(baseline_batch_factory, int(training["epochs"]))
    consumed_microbatches = int(restored_state.get("consumed_microbatches", global_step * accumulation_steps))
    if runtime_mode == "resume":
        train_batches = _resume_after_microbatches(train_batches, consumed_microbatches)
    microbatches_seen = consumed_microbatches
    pending_microbatches = 0
    running_mlm_loss = 0.0
    running_masked_tokens = 0
    running_masked_correct = 0
    running_processed_tokens = 0
    running_sequences = 0
    processed_tokens = int(restored_state.get("processed_tokens", 0))
    processed_sequences = int(restored_state.get("processed_sequences", 0))
    started_at = time.perf_counter()
    stop_training = False
    for epoch_index, batch in train_batches:
        if global_step >= total_steps:
            break
        model.train()
        inputs = _mask_batch(batch, tokenizer, float(training["mask_probability"]), int(training["seed"]) + global_step, device)
        autocast = torch.cuda.amp.autocast if use_fp16 else nullcontext
        with autocast():
            model_output = model(**inputs)
            raw_loss = model_output.loss
            if raw_loss is None:
                raise RuntimeError("Baseline MLM training did not produce a loss.")
            loss = raw_loss / accumulation_steps
        masked_positions = inputs["labels"] != -100
        running_masked_tokens += int(masked_positions.sum().item())
        running_masked_correct += int(
            (model_output.logits.argmax(dim=-1)[masked_positions] == inputs["labels"][masked_positions]).sum().item()
        )
        running_processed_tokens += int(inputs["attention_mask"].sum().item())
        running_sequences += len(batch)
        scaler.scale(loss).backward() if use_fp16 else loss.backward()
        pending_microbatches += 1
        microbatches_seen += 1
        running_mlm_loss += float(raw_loss.detach().cpu())
        if pending_microbatches < accumulation_steps:
            continue
        if use_fp16:
            scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clipping"])).detach().cpu())
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        writer.add_scalar("train/mlm_loss", running_mlm_loss / pending_microbatches, global_step)
        writer.add_scalar("train/gradient_norm", gradient_norm, global_step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
        writer.add_scalar("train/epoch", epoch_index + 1, global_step)
        processed_tokens += running_processed_tokens
        processed_sequences += running_sequences
        writer.add_scalar("train/masked_token_accuracy", running_masked_correct / max(1, running_masked_tokens), global_step)
        writer.add_scalar("train/processed_tokens", processed_tokens, global_step)
        writer.add_scalar("train/processed_sequences", processed_sequences, global_step)
        writer.add_scalar("train/elapsed_seconds", time.perf_counter() - started_at, global_step)
        if device.type == "cuda":
            writer.add_scalar("system/gpu_max_memory_bytes", torch.cuda.max_memory_allocated(device), global_step)
        pending_microbatches = 0
        running_mlm_loss = 0.0
        running_masked_tokens = 0
        running_masked_correct = 0
        running_processed_tokens = 0
        running_sequences = 0

        if global_step % int(training["evaluation_interval"]) == 0:
            per_language = {
                language: _evaluate(
                    model,
                    project_root,
                    plan_path,
                    tokenizer,
                    int(model_settings["max_sequence_length"]),
                    training,
                    language,
                    device,
                    global_step,
                )
                for language in ("en", "es", "hi")
            }
            validation_loss = sum(per_language.values()) / len(per_language)
            writer.add_scalar("validation/macro_mlm_loss", validation_loss, global_step)
            for language, value in per_language.items():
                writer.add_scalar(f"validation/{language}_mlm_loss", value, global_step)
            state = {
                "global_step": global_step,
                "macro_validation_loss": validation_loss,
                "best_macro_validation_loss": best_loss,
                "no_improvement": no_improvement,
                "consumed_microbatches": microbatches_seen,
                "processed_tokens": processed_tokens,
                "processed_sequences": processed_sequences,
                "per_language": per_language,
                "epoch": epoch_index + 1,
            }
            if validation_loss < best_loss:
                best_loss = validation_loss
                no_improvement = 0
                state["best_macro_validation_loss"] = best_loss
                state["no_improvement"] = no_improvement
                _save_checkpoint(destination / "best", model, tokenizer, optimizer, scheduler, state)
            else:
                no_improvement += 1
            if no_improvement >= int(training["early_stopping_patience"]):
                stop_training = True
        if global_step % int(training["checkpoint_interval"]) == 0:
            _save_checkpoint(
                destination / f"checkpoint-{global_step:08d}",
                model,
                tokenizer,
                optimizer,
                scheduler,
                {
                    "global_step": global_step,
                    "best_macro_validation_loss": best_loss,
                    "no_improvement": no_improvement,
                    "consumed_microbatches": microbatches_seen,
                    "processed_tokens": processed_tokens,
                    "processed_sequences": processed_sequences,
                    "epoch": epoch_index + 1,
                },
            )
            _prune_periodic_checkpoints(destination, int(training["maximum_retained_checkpoints"]))
        if stop_training:
            break

    if pending_microbatches:
        raise RuntimeError("Corpus ended mid-gradient-accumulation; rerun with a batch size or accumulation factor that yields full optimizer steps.")
    final_state = {
        "global_step": global_step,
        "best_macro_validation_loss": best_loss,
        "no_improvement": no_improvement,
        "consumed_microbatches": microbatches_seen,
        "processed_tokens": processed_tokens,
        "processed_sequences": processed_sequences,
        "epoch": epoch_index + 1 if "epoch_index" in locals() else 0,
    }
    _save_checkpoint(destination / "final", model, tokenizer, optimizer, scheduler, final_state)
    writer.close()
    selected_checkpoint = _resolve_checkpoint(destination, "best")
    selected_model = RobertaForMaskedLM.from_pretrained(str(selected_checkpoint)).to(device)
    test_metrics = {
        language: _evaluate(
            selected_model,
            project_root,
            plan_path,
            tokenizer,
            int(model_settings["max_sequence_length"]),
            training,
            language,
            device,
            int(training["seed"]),
            split="test",
        )
        for language in ("en", "es", "hi")
    }
    final_state.update(
        {
            "selected_checkpoint": str(selected_checkpoint),
            "test_per_language": test_metrics,
            "test_macro_mlm_loss": sum(test_metrics.values()) / len(test_metrics),
        }
    )
    _save_json_result(results_directory / "mlm" / f"{group_name}_test_metrics.json", final_state)
    record_group_result(experiment_directory, group_name, final_state)
    return final_state
