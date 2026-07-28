"""Language-conditioned, character-aligned candidate representation fusion."""

from __future__ import annotations

import torch
from torch import nn


class LanguageConditionedCandidateFusion(nn.Module):
    """Fuse candidate embeddings with priors, language routing, and a learned gate."""

    def __init__(self, hidden_size: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.candidate_gate = nn.Linear(hidden_size, 1)

    def forward(
        self,
        aligned_embeddings: torch.Tensor,
        candidate_prior_probabilities: torch.Tensor,
        candidate_language_evidence: torch.Tensor,
        routing_probabilities: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
        fusion_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused embeddings, effective candidate weights, and score logits.

        ``candidate_language_evidence`` is a per-candidate distribution over
        languages, supplied by the tokenizer's segmentation diagnostics.
        """

        if aligned_embeddings.ndim != 4:
            raise ValueError("aligned_embeddings must have shape [batch, candidates, tokens, hidden].")
        batch_size, candidate_count, _, _ = aligned_embeddings.shape
        expected = (batch_size, candidate_count)
        if candidate_prior_probabilities.shape != expected:
            raise ValueError("candidate_prior_probabilities must be [batch, candidates].")
        if candidate_language_evidence.shape[:2] != expected:
            raise ValueError("candidate_language_evidence must begin [batch, candidates].")
        if routing_probabilities.shape != (batch_size, candidate_language_evidence.shape[-1]):
            raise ValueError("routing_probabilities must align with language evidence.")

        prior_score = torch.log(candidate_prior_probabilities.clamp_min(torch.finfo(aligned_embeddings.dtype).eps))
        language_score = (candidate_language_evidence * routing_probabilities[:, None, :]).sum(dim=-1)
        if candidate_token_mask is None:
            candidate_summary = aligned_embeddings.mean(dim=2)
        else:
            if candidate_token_mask.shape != aligned_embeddings.shape[:3]:
                raise ValueError("candidate_token_mask must be [batch, candidates, tokens].")
            token_weights = candidate_token_mask.to(aligned_embeddings.dtype).unsqueeze(-1)
            candidate_summary = (aligned_embeddings * token_weights).sum(dim=2) / token_weights.sum(dim=2).clamp_min(1.0)
        learned_gate = self.candidate_gate(candidate_summary).squeeze(-1)
        score_logits = self.alpha * prior_score + self.beta * language_score + learned_gate
        if candidate_mask is not None:
            score_logits = score_logits.masked_fill(~candidate_mask.bool(), torch.finfo(score_logits.dtype).min)
        if fusion_enabled:
            weights = torch.softmax(score_logits, dim=1)
        else:
            weights = torch.zeros_like(score_logits)
            weights[:, 0] = 1.0
        fused = (aligned_embeddings * weights[:, :, None, None]).sum(dim=1)
        return fused, weights, score_logits
