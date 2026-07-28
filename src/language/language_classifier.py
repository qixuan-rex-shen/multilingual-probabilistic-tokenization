"""Small differentiable language router used by the fused MLM path.

This is intentionally not a standalone large language-identification model.  It
learns a distribution over the three pretraining-corpus languages from the
reference candidate's token embeddings, allowing MLM loss gradients to adjust
the routing decision used during candidate fusion.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Final, Iterator

import torch
from torch import nn

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from src.data.pipeline import iter_prepared_corpus

LANGUAGES: Final[tuple[str, str, str]] = ("en", "es", "hi")
LANGUAGE_TO_ID: Final[dict[str, int]] = {language: index for index, language in enumerate(LANGUAGES)}


class LanguageRoutingClassifier(nn.Module):
    """A compact embedding-pooling classifier returning ``P(language | text)``."""

    def __init__(self, hidden_size: int, num_languages: int = len(LANGUAGES), dropout: float = 0.1) -> None:
        super().__init__()
        reduced_size = max(32, hidden_size // 4)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, reduced_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_size, num_languages),
        )

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and probabilities from ``[batch, tokens, hidden]`` inputs."""

        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [batch, tokens, hidden].")
        if attention_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("attention_mask must align with token_embeddings.")
        weights = attention_mask.to(token_embeddings.dtype).unsqueeze(-1)
        pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.network(pooled)
        return logits, torch.softmax(logits, dim=-1)


class CulturaXLanguagePriorClassifier:
    """Frozen fastText-style character n-gram classifier for candidate selection.

    It is trained only from ``(CulturaX text, language)`` pairs.  This small
    CPU model is intentionally separate from the differentiable language router
    inside the fused MLM: it supplies ``P(L|X)`` before candidates are chosen;
    the neural router supplies differentiable routing weights during fusion.
    """

    artifact_filename: Final[str] = "language_prior_classifier.pkl"
    metadata_filename: Final[str] = "training_metadata.json"

    def __init__(self, vectorizer: HashingVectorizer, classifier: SGDClassifier) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_probabilities(self, text: str) -> list[float]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        result = [0.0] * len(LANGUAGES)
        for class_id, probability in zip(self.classifier.classes_, probabilities):
            result[int(class_id)] = float(probability)
        total = sum(result)
        if total <= 0:
            raise RuntimeError("Language classifier returned an invalid probability distribution.")
        return [value / total for value in result]

    def save(self, destination: Path, metadata: dict[str, Any]) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / self.artifact_filename).open("wb") as artifact_file:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, artifact_file)
        (destination / self.metadata_filename).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, destination: Path) -> "CulturaXLanguagePriorClassifier":
        artifact_path = destination / cls.artifact_filename
        metadata_path = destination / cls.metadata_filename
        if not artifact_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete language-classifier artifact: {destination}")
        with artifact_path.open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
        return cls(payload["vectorizer"], payload["classifier"])


def _language_classifier_identity(plan_path: Path, settings: dict[str, Any]) -> str:
    payload = {
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "settings": settings,
        "languages": LANGUAGES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _iter_classifier_examples(
    project_root: Path,
    plan_path: Path,
    maximum_per_language: int,
) -> Iterator[tuple[str, int]]:
    """Stream an equal number of training-only examples from each language."""

    seen = {language: 0 for language in LANGUAGES}
    for record in iter_prepared_corpus(project_root, plan_path, split="train", balanced=True):
        language = record["language"]
        if language not in LANGUAGE_TO_ID or seen[language] >= maximum_per_language:
            continue
        seen[language] += 1
        yield record["text"], LANGUAGE_TO_ID[language]
        if all(count >= maximum_per_language for count in seen.values()):
            return


def train_culturax_language_prior_classifier(
    project_root: Path,
    plan_path: Path,
    settings: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Train or validate the small language-prior classifier atomically."""

    required = {"algorithm", "max_examples_per_language", "ngram_min", "ngram_max", "hash_features", "batch_size"}
    missing = required.difference(settings)
    if missing:
        raise KeyError(f"language_classifier configuration is missing: {sorted(missing)}")
    if settings["algorithm"] != "char_ngram_sgd":
        raise ValueError("Only language_classifier.algorithm: char_ngram_sgd is currently supported.")
    identity = _language_classifier_identity(plan_path, settings)
    metadata_path = destination / CulturaXLanguagePriorClassifier.metadata_filename
    artifact_path = destination / CulturaXLanguagePriorClassifier.artifact_filename
    if destination.exists():
        if metadata_path.is_file() and artifact_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("identity_sha256") != identity:
                raise ValueError(f"Language-classifier artifact does not match current corpus/configuration: {destination}")
            CulturaXLanguagePriorClassifier.load(destination)
            return metadata
        if destination.is_dir() and {entry.name for entry in destination.iterdir()} == {".gitkeep"}:
            print(f"Removing empty language-classifier scaffold: {destination}")
            (destination / ".gitkeep").unlink()
            destination.rmdir()
        else:
            raise FileExistsError(f"Incomplete language-classifier artifact: {destination}")
    temporary_destination = destination.with_name(f"{destination.name}.incomplete")
    if temporary_destination.exists():
        raise FileExistsError(f"Incomplete language-classifier artifact exists: {temporary_destination}")

    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(int(settings["ngram_min"]), int(settings["ngram_max"])),
        n_features=int(settings["hash_features"]),
        alternate_sign=False,
        norm="l2",
    )
    classifier = SGDClassifier(loss="log_loss", alpha=float(settings.get("alpha", 1e-5)), random_state=int(settings["seed"]))
    batch_text: list[str] = []
    batch_labels: list[int] = []
    examples_by_language = {language: 0 for language in LANGUAGES}
    first_batch = True
    for text, label in _iter_classifier_examples(
        project_root, plan_path, int(settings["max_examples_per_language"])
    ):
        batch_text.append(text)
        batch_labels.append(label)
        examples_by_language[LANGUAGES[label]] += 1
        if len(batch_text) < int(settings["batch_size"]):
            continue
        classifier.partial_fit(
            vectorizer.transform(batch_text), batch_labels, classes=list(range(len(LANGUAGES))) if first_batch else None
        )
        first_batch = False
        batch_text.clear()
        batch_labels.clear()
    if batch_text:
        classifier.partial_fit(
            vectorizer.transform(batch_text), batch_labels, classes=list(range(len(LANGUAGES))) if first_batch else None
        )
        first_batch = False
    if first_batch or any(count == 0 for count in examples_by_language.values()):
        raise ValueError("No balanced CulturaX training examples were available for language-classifier fitting.")

    metadata = {
        "kind": "culturax_char_ngram_language_prior",
        "identity_sha256": identity,
        "plan_path": str(plan_path),
        "settings": settings,
        "languages": list(LANGUAGES),
        "examples_by_language": examples_by_language,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {temporary_destination}")
    CulturaXLanguagePriorClassifier(vectorizer, classifier).save(temporary_destination, metadata)
    temporary_destination.replace(destination)
    return metadata
