"""Matched local GLueCoS fine-tuning for BPE and fused-tokenization models.

The two downstream arms deliberately share this module's task head, optimizer,
schedule, seed protocol, split handling, and checkpoint selection rule.  The
only arm-specific operation is how the transformer input embedding is formed:
the BPE control uses one deterministic token sequence, whereas the proposed
arm projects and fuses language-conditioned candidate tokenizations.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM

from src.data.gluecos_loader import infer_task, safe_dataset_name
from src.evaluation.metrics import classification_metrics, token_classification_metrics
from src.language.language_classifier import CulturaXLanguagePriorClassifier
from src.models.xlmr import LanguageConditionedFusedRobertaForMaskedLM
from src.tokenizer.alignment import project_candidate_embeddings_to_reference
from src.tokenizer.common import build_xlmr_single_sequence
from src.tokenizer.probabilistic import CandidateTokenization, UnigramCandidateTokenizer


@dataclass
class EncodedExample:
    """One local GLueCoS example after either tokenizer has encoded it."""

    input_ids: list[int] | None
    attention_mask: list[int] | None
    candidates: list[CandidateTokenization] | None
    label: int | list[int]


@dataclass
class TaskOutput:
    """A lightweight common output shape for both downstream arms."""

    loss: torch.Tensor | None
    logits: torch.Tensor


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a concise epoch record under the configured log directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _word_spans(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for word in words:
        spans.append((position, position + len(word)))
        position += len(word) + 1
    return " ".join(words), spans


def _token_labels(
    offsets: list[tuple[int, int]], word_spans: list[tuple[int, int]], word_labels: list[int]
) -> list[int]:
    """Attach each original-word label to its first overlapping subword only."""

    labels: list[int] = []
    emitted_words: set[int] = set()
    for start, end in offsets:
        if start < 0 or end <= start:
            labels.append(-100)
            continue
        matching = next(
            (
                index
                for index, (word_start, word_end) in enumerate(word_spans)
                if start < word_end and end > word_start
            ),
            None,
        )
        if matching is None or matching in emitted_words:
            labels.append(-100)
        else:
            labels.append(word_labels[matching])
            emitted_words.add(matching)
    return labels


def _dataset_splits(dataset: Any) -> tuple[str, str, str]:
    names = list(dataset.keys())
    train = next((name for name in names if name.lower().startswith("train")), None)
    validation = next(
        (name for name in names if name.lower().startswith(("dev", "validation", "valid"))), None
    )
    test = next((name for name in names if name.lower().startswith("test")), None)
    if not train or not validation or not test:
        raise ValueError(f"Expected train/dev/test splits, received {names}")
    return train, validation, test


def _limited_rows(split: Any, limit: int) -> list[dict[str, Any]]:
    """Return a deterministic prefix only when an explicit smoke limit is set."""

    if limit < 0:
        raise ValueError("finetuning.max_examples_per_split must be zero or positive.")
    count = len(split) if limit == 0 else min(len(split), limit)
    return [split[index] for index in range(count)]


def _label_field(task: str) -> str:
    return {"ner": "ner_tags", "pos": "pos_tags_secondary", "sentiment": "sentiment_label"}[task]


def _labels_for_dataset(dataset: Any, task: str) -> dict[str, int]:
    """Build one deterministic label vocabulary from every fixed split."""

    field = _label_field(task)
    values: set[str] = set()
    for split in dataset.values():
        for label in split[field]:
            if isinstance(label, list):
                values.update(str(value) for value in label)
            else:
                values.add(str(label))
    if not values:
        raise ValueError(f"No labels were found for {task}.")
    return {label: index for index, label in enumerate(sorted(values))}


def _encode_bpe_example(
    example: dict[str, Any],
    task: str,
    labels: dict[str, int],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> EncodedExample:
    if task == "sentiment":
        text = str(example["text"])
        gold: int | list[int] = labels[str(example[_label_field(task)])]
    else:
        words = [str(word) for word in example["tokens"]]
        text, spans = _word_spans(words)
        gold_words = [labels[str(value)] for value in example[_label_field(task)]]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length - 2,
    )
    content_ids = list(encoded["input_ids"])
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    input_ids = build_xlmr_single_sequence(tokenizer, content_ids)
    full_offsets = [(-1, -1), *offsets, (-1, -1)]
    if task != "sentiment":
        gold = _token_labels(full_offsets, spans, gold_words)
    return EncodedExample(input_ids=input_ids, attention_mask=[1] * len(input_ids), candidates=None, label=gold)


def _encode_fused_example(
    example: dict[str, Any],
    task: str,
    labels: dict[str, int],
    tokenizer: UnigramCandidateTokenizer,
    language_prior: CulturaXLanguagePriorClassifier,
    max_length: int,
) -> EncodedExample:
    if task == "sentiment":
        text = str(example["text"])
        gold: int | list[int] = labels[str(example[_label_field(task)])]
    else:
        words = [str(word) for word in example["tokens"]]
        text, spans = _word_spans(words)
        gold_words = [labels[str(value)] for value in example[_label_field(task)]]
    candidates = tokenizer.encode_candidates(
        text,
        max_sequence_length=max_length,
        language_probabilities=language_prior.predict_probabilities(text),
    )
    if not candidates:
        raise RuntimeError("No probabilistic candidates were available for a downstream example.")
    if task != "sentiment":
        gold = _token_labels(candidates[0].offsets, spans, gold_words)
    return EncodedExample(input_ids=None, attention_mask=None, candidates=candidates, label=gold)


def _pad_bpe(examples: list[EncodedExample], task: str, device: torch.device) -> dict[str, torch.Tensor]:
    length = max(len(example.input_ids or []) for example in examples)
    ids = torch.ones((len(examples), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = (
        torch.full((len(examples), length), -100, dtype=torch.long)
        if task != "sentiment"
        else torch.empty(len(examples), dtype=torch.long)
    )
    for row, example in enumerate(examples):
        sequence = example.input_ids or []
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
        if task == "sentiment":
            labels[row] = int(example.label)
        else:
            labels[row, : len(example.label)] = torch.tensor(example.label, dtype=torch.long)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def _pad_fused(examples: list[EncodedExample], task: str, device: torch.device) -> dict[str, torch.Tensor]:
    candidate_count = max(len(example.candidates or []) for example in examples)
    length = max(
        len(candidate.input_ids) for example in examples for candidate in (example.candidates or [])
    )
    batch = len(examples)
    ids = torch.ones((batch, candidate_count, length), dtype=torch.long)
    attention = torch.zeros_like(ids)
    spans = torch.full((batch, candidate_count, length, 2), -1, dtype=torch.long)
    priors = torch.zeros((batch, candidate_count), dtype=torch.float)
    evidence = torch.zeros((batch, candidate_count, 3), dtype=torch.float)
    candidate_mask = torch.zeros((batch, candidate_count), dtype=torch.bool)
    labels = (
        torch.full((batch, length), -100, dtype=torch.long)
        if task != "sentiment"
        else torch.empty(batch, dtype=torch.long)
    )
    for row, example in enumerate(examples):
        for index, candidate in enumerate(example.candidates or []):
            size = len(candidate.input_ids)
            ids[row, index, :size] = torch.tensor(candidate.input_ids, dtype=torch.long)
            attention[row, index, :size] = 1
            spans[row, index, :size] = torch.tensor(candidate.offsets, dtype=torch.long)
            priors[row, index] = float(candidate.prior_probability)
            evidence[row, index] = torch.tensor(candidate.language_evidence, dtype=torch.float)
            candidate_mask[row, index] = True
        if task == "sentiment":
            labels[row] = int(example.label)
        else:
            labels[row, : len(example.label)] = torch.tensor(example.label, dtype=torch.long)
    return {
        "candidate_input_ids": ids.to(device),
        "candidate_attention_mask": attention.to(device),
        "candidate_char_spans": spans.to(device),
        "candidate_prior_probabilities": priors.to(device),
        "candidate_language_evidence": evidence.to(device),
        "candidate_mask": candidate_mask.to(device),
        "labels": labels.to(device),
    }


class StandardTaskHead(nn.Module):
    """The exact same head is used for BPE and proposed downstream models."""

    def __init__(self, hidden_size: int, dropout_probability: float, num_labels: int, task: str) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_probability)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.task = task

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        states = hidden_states[:, 0] if self.task == "sentiment" else hidden_states
        states = self.dropout(states)
        states = torch.tanh(self.dense(states))
        return self.out_proj(self.dropout(states))


def _task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


class BPETaskModel(nn.Module):
    """Clearly named deterministic single-tokenization downstream baseline."""

    def __init__(self, pretrained: RobertaForMaskedLM, num_labels: int, task: str) -> None:
        super().__init__()
        self.roberta = pretrained.roberta
        self.task_head = StandardTaskHead(
            pretrained.config.hidden_size, pretrained.config.hidden_dropout_prob, num_labels, task
        )

    def forward(self, labels: torch.Tensor | None = None, **inputs: torch.Tensor) -> TaskOutput:
        hidden = self.roberta(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True
        ).last_hidden_state
        logits = self.task_head(hidden)
        return TaskOutput(_task_loss(logits, labels) if labels is not None else None, logits)


class FusedTaskModel(nn.Module):
    """Downstream model retaining the proposed fusion path into ``inputs_embeds``."""

    def __init__(self, pretrained: LanguageConditionedFusedRobertaForMaskedLM, num_labels: int, task: str) -> None:
        super().__init__()
        self.roberta = pretrained.base_model.roberta
        self.language_classifier = pretrained.language_classifier
        self.fusion = pretrained.fusion
        self.task_head = StandardTaskHead(
            pretrained.config.hidden_size, pretrained.config.hidden_dropout_prob, num_labels, task
        )

    def forward(self, labels: torch.Tensor | None = None, **inputs: torch.Tensor) -> TaskOutput:
        ids = inputs["candidate_input_ids"]
        attention = inputs["candidate_attention_mask"]
        candidate_spans = inputs["candidate_char_spans"]
        word_embeddings = self.roberta.embeddings.word_embeddings(ids)
        reference = word_embeddings[:, 0]
        reference_attention = attention[:, 0]
        aligned = project_candidate_embeddings_to_reference(
            word_embeddings, candidate_spans, candidate_spans[:, 0], reference
        )
        _, routing = self.language_classifier(reference, reference_attention)
        fused, _, _ = self.fusion(
            aligned,
            inputs["candidate_prior_probabilities"],
            inputs["candidate_language_evidence"],
            routing,
            inputs["candidate_mask"],
            attention,
            True,
        )
        hidden = self.roberta(
            input_ids=None,
            inputs_embeds=fused,
            attention_mask=reference_attention,
            return_dict=True,
        ).last_hidden_state
        logits = self.task_head(hidden)
        return TaskOutput(_task_loss(logits, labels) if labels is not None else None, logits)


def _initialize_shared_task_head(model: nn.Module, seed: int, initializer_range: float) -> None:
    """Give both arms bit-identical task-head initialization for each seed."""

    _set_seed(seed + 100_003)
    head = getattr(model, "task_head")
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _load_bpe_task_model(checkpoint: Path, task: str, labels: int, device: torch.device) -> BPETaskModel:
    return BPETaskModel(RobertaForMaskedLM.from_pretrained(str(checkpoint)), labels, task).to(device)


def _batches(examples: list[EncodedExample], batch_size: int, seed: int, shuffle: bool) -> Iterator[list[EncodedExample]]:
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def _evaluate(
    model: nn.Module,
    examples: list[EncodedExample],
    task: str,
    fused: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    predicted: list[int] = []
    expected: list[int] = []
    with torch.no_grad():
        for rows in _batches(examples, batch_size, 0, False):
            inputs = _pad_fused(rows, task, device) if fused else _pad_bpe(rows, task, device)
            labels = inputs.pop("labels")
            output = model(labels=labels, **inputs)
            if task == "sentiment":
                predicted.extend(output.logits.argmax(-1).cpu().tolist())
                expected.extend(labels.cpu().tolist())
            else:
                valid = labels != -100
                predicted.extend(output.logits.argmax(-1)[valid].cpu().tolist())
                expected.extend(labels[valid].cpu().tolist())
    metrics = (
        classification_metrics(predicted, expected)
        if task == "sentiment"
        else token_classification_metrics([predicted], [expected])
    )
    return metrics, predicted, expected


def _train_one_arm(
    task: str,
    arm: str,
    train: list[EncodedExample],
    validation: list[EncodedExample],
    test: list[EncodedExample],
    labels: dict[str, int],
    seed: int,
    settings: dict[str, Any],
    checkpoint: Path,
    log_path: Path,
    device: torch.device,
    bpe_checkpoint: Path | None = None,
    fused_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run one seed and one arm using matched training settings."""

    _set_seed(seed)
    fused = arm == "probabilistic"
    if fused:
        if fused_checkpoint is None:
            raise ValueError("Missing probabilistic MLM checkpoint.")
        model: nn.Module = FusedTaskModel(
            LanguageConditionedFusedRobertaForMaskedLM.from_pretrained(fused_checkpoint), len(labels), task
        ).to(device)
    else:
        if bpe_checkpoint is None:
            raise ValueError("Missing BPE MLM checkpoint.")
        model = _load_bpe_task_model(bpe_checkpoint, task, len(labels), device)
    _initialize_shared_task_head(model, seed, float(getattr(model, "roberta").config.initializer_range))

    if checkpoint.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing selected fine-tuning checkpoint: {checkpoint}. "
            "Use evaluation mode or archive the old result first."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    batches_per_epoch = math.ceil(len(train) / int(settings["batch_size"]))
    total_updates = max(1, int(settings["epochs"]) * batches_per_epoch)
    warmup_updates = int(round(total_updates * float(settings["warmup_ratio"])))

    def learning_rate_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        return max(0.0, float(total_updates - step) / float(max(1, total_updates - warmup_updates)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    selection_metric = str(settings["checkpoint_selection_metric"])
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    epoch_history: list[dict[str, Any]] = []

    for epoch in range(int(settings["epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for rows in _batches(train, int(settings["batch_size"]), seed + epoch, True):
            inputs = _pad_fused(rows, task, device) if fused else _pad_bpe(rows, task, device)
            labels_tensor = inputs.pop("labels")
            output = model(labels=labels_tensor, **inputs)
            if output.loss is None:
                raise RuntimeError("Fine-tuning loss was not produced.")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clipping"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(output.loss.detach().cpu()))
        validation_metrics, _, _ = _evaluate(
            model, validation, task, fused, int(settings["batch_size"]), device
        )
        record = {
            "arm": arm,
            "task": task,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": validation_metrics,
        }
        _append_jsonl(log_path, record)
        epoch_history.append(record)
        print(
            f"{task}/{arm}/seed={seed}/epoch={epoch + 1}: "
            f"validation {selection_metric}={validation_metrics[selection_metric]:.4f}"
        )
        if validation_metrics[selection_metric] > best_score:
            best_score = validation_metrics[selection_metric]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a selected checkpoint.")
    print(f"Saving checkpoint to: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp")
    torch.save(
        {"state_dict": best_state, "labels": labels, "task": task, "seed": seed, "history": epoch_history},
        temporary,
    )
    temporary.replace(checkpoint)
    model.load_state_dict(best_state)
    metrics, predictions, expected = _evaluate(model, test, task, fused, int(settings["batch_size"]), device)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "labels": expected,
        "validation_metric": selection_metric,
        "validation_score": best_score,
        "checkpoint": str(checkpoint),
        "history": epoch_history,
    }


def run_gluecos_finetuning(
    project_root: Path,
    config: dict[str, Any],
    bpe_mlm_checkpoint: Path,
    fused_mlm_checkpoint: Path,
) -> dict[str, Any]:
    """Fine-tune both saved MLM arms across all configured local GLueCoS tasks."""

    from datasets import load_from_disk

    mode = str(config["runtime"]["mode"])
    if mode not in {"train", "resume", "evaluate"}:
        raise ValueError(f"Unsupported runtime mode: {mode}")
    settings = config["finetuning"]
    if settings["optimizer"] != "AdamW" or settings["scheduler"] != "linear_warmup":
        raise ValueError("This fine-tuning implementation supports AdamW with linear_warmup only.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(project_root / config["paths"]["tokenizers"] / "bpe")
    )
    proposed_tokenizer = UnigramCandidateTokenizer.from_pretrained(
        project_root / config["paths"]["tokenizers"] / "probabilistic"
    )
    proposed_tokenizer.configure_candidate_selection(config["probabilistic_tokenizer"])
    language_prior = CulturaXLanguagePriorClassifier.load(
        project_root / config["paths"]["language_classifier"]
    )
    all_results: dict[str, Any] = {}
    example_limit = int(settings.get("max_examples_per_split", 0))

    for repository in config["data_download"]["gluecos"]:
        task = infer_task(repository)
        dataset_name = safe_dataset_name(repository)
        dataset = load_from_disk(
            str(project_root / config["paths"]["processed_data"] / "downstream_tasks" / dataset_name)
        )
        train_name, validation_name, test_name = _dataset_splits(dataset)
        labels = _labels_for_dataset(dataset, task)
        encode_bpe = lambda row: _encode_bpe_example(
            row, task, labels, bpe_tokenizer, int(config["model"]["max_sequence_length"])
        )
        encode_fused = lambda row: _encode_fused_example(
            row,
            task,
            labels,
            proposed_tokenizer,
            language_prior,
            int(config["model"]["max_sequence_length"]),
        )
        encoded = {
            "bpe": {
                "train": [encode_bpe(row) for row in _limited_rows(dataset[train_name], example_limit)],
                "validation": [encode_bpe(row) for row in _limited_rows(dataset[validation_name], example_limit)],
                "test": [encode_bpe(row) for row in _limited_rows(dataset[test_name], example_limit)],
            },
            "probabilistic": {
                "train": [encode_fused(row) for row in _limited_rows(dataset[train_name], example_limit)],
                "validation": [encode_fused(row) for row in _limited_rows(dataset[validation_name], example_limit)],
                "test": [encode_fused(row) for row in _limited_rows(dataset[test_name], example_limit)],
            },
        }
        task_results: dict[str, Any] = {
            "task": task,
            "labels": labels,
            "splits": {"train": train_name, "validation": validation_name, "test": test_name},
            "max_examples_per_split": example_limit,
            "seeds": {},
        }
        for configured_seed in config["evaluation"]["seeds"]:
            seed = int(configured_seed)
            destination = project_root / config["paths"]["results"] / task / dataset_name / f"seed_{seed}.json"
            if destination.is_file():
                seed_results = json.loads(destination.read_text(encoding="utf-8"))
                if set(seed_results) == {"bpe", "probabilistic"}:
                    task_results["seeds"][str(seed)] = seed_results
                    continue
                raise ValueError(f"Existing downstream result is malformed: {destination}")
            if mode == "evaluate":
                raise FileNotFoundError(f"Evaluation requested but no saved result is available: {destination}")
            seed_results: dict[str, Any] = {}
            for arm in ("bpe", "probabilistic"):
                checkpoint = (
                    project_root
                    / config["paths"]["downstream_checkpoints"]
                    / arm
                    / dataset_name
                    / f"seed_{seed}"
                    / "best.pt"
                )
                log_path = (
                    project_root
                    / config["paths"]["logs"]
                    / "downstream"
                    / task
                    / dataset_name
                    / f"seed_{seed}_{arm}.jsonl"
                )
                seed_results[arm] = _train_one_arm(
                    task,
                    arm,
                    encoded[arm]["train"],
                    encoded[arm]["validation"],
                    encoded[arm]["test"],
                    labels,
                    seed,
                    settings,
                    checkpoint,
                    log_path,
                    device,
                    bpe_mlm_checkpoint,
                    fused_mlm_checkpoint,
                )
            print(f"Saving checkpoint to: {destination}")
            _atomic_json(destination, seed_results)
            task_results["seeds"][str(seed)] = seed_results
        all_results[repository] = task_results

    experiment_directory = project_root / config["paths"]["experiments"] / config["experiment"]["id"]
    summary_path = experiment_directory / "downstream_metrics.json"
    print(f"Saving checkpoint to: {summary_path}")
    _atomic_json(summary_path, all_results)
    return all_results
