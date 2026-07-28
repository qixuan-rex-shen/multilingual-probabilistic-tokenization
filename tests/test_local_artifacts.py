"""Focused checks for restart-safe tokenizer outputs and language priors."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language.language_classifier import CulturaXLanguagePriorClassifier


def run_local_artifact_tests() -> None:
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 3), n_features=256, alternate_sign=False)
    classifier = SGDClassifier(loss="log_loss", random_state=0)
    texts = ["the president spoke", "el presidente hablo", "राष्ट्रपति ने कहा"]
    labels = [0, 1, 2]
    classifier.partial_fit(vectorizer.transform(texts), labels, classes=[0, 1, 2])
    prior = CulturaXLanguagePriorClassifier(vectorizer, classifier).predict_probabilities("El presidente habló")
    assert len(prior) == 3 and abs(sum(prior) - 1.0) < 1e-6


if __name__ == "__main__":
    run_local_artifact_tests()
    print("Local artifact and CulturaX language-prior tests passed.")
