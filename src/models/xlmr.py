"""Randomly initialized XLM-R-style MLMs, including the fused proposed model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from transformers import RobertaConfig, RobertaForMaskedLM

from src.language.language_classifier import LanguageRoutingClassifier
from src.models.fusion import LanguageConditionedCandidateFusion
from src.tokenizer.alignment import project_candidate_embeddings_to_reference


def _roberta_config(model_settings: dict[str, Any], vocab_size: int) -> RobertaConfig:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=int(model_settings["hidden_size"]),
        num_hidden_layers=int(model_settings["layers"]),
        num_attention_heads=int(model_settings["attention_heads"]),
        intermediate_size=int(model_settings["intermediate_size"]),
        hidden_dropout_prob=float(model_settings["hidden_dropout"]),
        attention_probs_dropout_prob=float(model_settings["attention_dropout"]),
        layer_norm_eps=float(model_settings["layer_norm_epsilon"]),
        max_position_embeddings=int(model_settings["max_position_embeddings"]),
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
    )


def build_xlmr_mlm(model_settings: dict[str, Any], vocab_size: int) -> RobertaForMaskedLM:
    """Create the clearly named single-tokenization BPE baseline from scratch."""

    return RobertaForMaskedLM(_roberta_config(model_settings, vocab_size))


@dataclass
class FusedMLMOutput:
    """Forward result for the proposed model with auditable routing information."""

    loss: torch.Tensor | None
    mlm_loss: torch.Tensor | None
    routing_loss: torch.Tensor | None
    logits: torch.Tensor
    candidate_weights: torch.Tensor
    routing_logits: torch.Tensor
    routing_probabilities: torch.Tensor
    fusion_score_logits: torch.Tensor
    fused_embeddings: torch.Tensor
    used_inputs_embeds: bool


class LanguageConditionedFusedRobertaForMaskedLM(nn.Module):
    """Proposed MLM: top-k paths -> aligned fusion -> RoBERTa ``inputs_embeds``.

    The internal RoBERTa module and MLM head have the same randomly initialized
    configuration as the baseline.  Only the input construction differs.
    """

    def __init__(
        self,
        config: RobertaConfig,
        alpha: float = 1.0,
        beta: float = 1.0,
        routing_auxiliary_loss_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = RobertaForMaskedLM(config)
        self.language_classifier = LanguageRoutingClassifier(
            config.hidden_size, dropout=config.hidden_dropout_prob
        )
        self.fusion = LanguageConditionedCandidateFusion(config.hidden_size, alpha=alpha, beta=beta)
        self.routing_auxiliary_loss_weight = float(routing_auxiliary_loss_weight)

    @property
    def config(self) -> RobertaConfig:
        return self.base_model.config

    def forward(
        self,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_char_spans: torch.Tensor,
        candidate_prior_probabilities: torch.Tensor,
        candidate_language_evidence: torch.Tensor,
        candidate_alignment_reference_indices: torch.Tensor | None = None,
        candidate_alignment_candidate_indices: torch.Tensor | None = None,
        candidate_alignment_overlap_weights: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        language_labels: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        fusion_enabled: bool = True,
    ) -> FusedMLMOutput:
        """Run candidate fusion and MLM prediction.

        Candidate 0 is the reference sequence.  All candidate token embeddings
        are projected onto its positions by their original-character spans, then
        fused.  The fused result is supplied to the transformer exclusively via
        ``inputs_embeds``; no selected candidate ``input_ids`` are passed to it.
        """

        if candidate_input_ids.ndim != 3:
            raise ValueError("candidate_input_ids must be [batch, candidates, tokens].")
        if candidate_attention_mask.shape != candidate_input_ids.shape:
            raise ValueError("candidate_attention_mask must match candidate_input_ids.")
        batch_size, candidate_count, candidate_length = candidate_input_ids.shape
        if candidate_char_spans.shape != (batch_size, candidate_count, candidate_length, 2):
            raise ValueError("candidate_char_spans must be [batch, candidates, tokens, 2].")
        if candidate_mask is None:
            candidate_mask = candidate_attention_mask.any(dim=-1)
        if candidate_mask.shape != (batch_size, candidate_count):
            raise ValueError("candidate_mask must be [batch, candidates].")

        word_embeddings = self.base_model.roberta.embeddings.word_embeddings(candidate_input_ids)
        reference_embeddings = word_embeddings[:, 0]
        reference_attention_mask = candidate_attention_mask[:, 0]
        reference_spans = candidate_char_spans[:, 0]
        aligned_embeddings = project_candidate_embeddings_to_reference(
            word_embeddings,
            candidate_char_spans,
            reference_spans,
            reference_embeddings,
            sparse_reference_indices=candidate_alignment_reference_indices,
            sparse_candidate_indices=candidate_alignment_candidate_indices,
            sparse_overlap_weights=candidate_alignment_overlap_weights,
        )
        routing_logits, routing_probabilities = self.language_classifier(
            reference_embeddings, reference_attention_mask
        )
        fused_embeddings, candidate_weights, fusion_score_logits = self.fusion(
            aligned_embeddings=aligned_embeddings,
            candidate_prior_probabilities=candidate_prior_probabilities,
            candidate_language_evidence=candidate_language_evidence,
            routing_probabilities=routing_probabilities,
            candidate_mask=candidate_mask,
            candidate_token_mask=candidate_attention_mask,
            fusion_enabled=fusion_enabled,
        )

        transformer_outputs = self.base_model.roberta(
            input_ids=None,
            inputs_embeds=fused_embeddings,
            attention_mask=reference_attention_mask,
            return_dict=True,
        )
        logits = self.base_model.lm_head(transformer_outputs.last_hidden_state)
        mlm_loss = None
        routing_loss = None
        loss = None
        if labels is not None:
            mlm_loss = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100
            )
            loss = mlm_loss
        if language_labels is not None:
            routing_loss = functional.cross_entropy(routing_logits, language_labels)
            loss = routing_loss * self.routing_auxiliary_loss_weight if loss is None else (
                loss + routing_loss * self.routing_auxiliary_loss_weight
            )
        return FusedMLMOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            routing_loss=routing_loss,
            logits=logits,
            candidate_weights=candidate_weights,
            routing_logits=routing_logits,
            routing_probabilities=routing_probabilities,
            fusion_score_logits=fusion_score_logits,
            fused_embeddings=fused_embeddings,
            used_inputs_embeds=True,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save base MLM plus the separate trainable routing/fusion modules."""

        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(destination))
        torch.save(
            {
                "language_classifier": self.language_classifier.state_dict(),
                "fusion": self.fusion.state_dict(),
            },
            destination / "candidate_fusion_state.pt",
        )
        (destination / "candidate_fusion_config.json").write_text(
            json.dumps(
                {
                    "alpha": self.fusion.alpha,
                    "beta": self.fusion.beta,
                    "routing_auxiliary_loss_weight": self.routing_auxiliary_loss_weight,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path, map_location: str | torch.device = "cpu") -> "LanguageConditionedFusedRobertaForMaskedLM":
        """Restore a saved fused model checkpoint."""

        source = Path(directory)
        settings = json.loads((source / "candidate_fusion_config.json").read_text(encoding="utf-8"))
        base_model = RobertaForMaskedLM.from_pretrained(str(source))
        model = cls(base_model.config, **settings)
        model.base_model.load_state_dict(base_model.state_dict())
        state = torch.load(source / "candidate_fusion_state.pt", map_location=map_location, weights_only=True)
        model.language_classifier.load_state_dict(state["language_classifier"])
        model.fusion.load_state_dict(state["fusion"])
        return model


def build_language_conditioned_fused_xlmr_mlm(
    model_settings: dict[str, Any],
    vocab_size: int,
    probabilistic_settings: dict[str, Any],
) -> LanguageConditionedFusedRobertaForMaskedLM:
    """Create the proposed top-k fusion MLM with no pretrained weights."""

    return LanguageConditionedFusedRobertaForMaskedLM(
        _roberta_config(model_settings, vocab_size),
        alpha=float(probabilistic_settings["alpha"]),
        beta=float(probabilistic_settings["beta"]),
        routing_auxiliary_loss_weight=float(probabilistic_settings.get("routing_auxiliary_loss_weight", 0.05)),
    )


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
