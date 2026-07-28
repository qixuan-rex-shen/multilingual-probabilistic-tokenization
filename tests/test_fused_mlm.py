"""Controlled forward/gradient checks for the language-conditioned fused MLM.

Run directly without a test runner:
    python tests/test_fused_mlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import RobertaConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xlmr import LanguageConditionedFusedRobertaForMaskedLM
from src.tokenizer.alignment import project_candidate_embeddings_to_reference
from src.training.pretrain import _build_sparse_alignment_edges


def _tiny_model() -> LanguageConditionedFusedRobertaForMaskedLM:
    torch.manual_seed(7)
    config = RobertaConfig(
        vocab_size=31,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return LanguageConditionedFusedRobertaForMaskedLM(config, alpha=1.0, beta=2.0, routing_auxiliary_loss_weight=0.0)


def _inputs() -> dict[str, torch.Tensor]:
    # Candidate 0: [0:2], [2:4], [4:6]; candidate 1: [0:3], [3:6].
    # Alignment is therefore by original character spans, not token positions.
    candidate_input_ids = torch.tensor(
        [
            [[0, 4, 5, 6, 2], [0, 7, 8, 2, 1]],
            [[0, 9, 10, 11, 2], [0, 12, 13, 2, 1]],
        ],
        dtype=torch.long,
    )
    candidate_attention_mask = torch.tensor(
        [
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
        ],
        dtype=torch.long,
    )
    candidate_char_spans = torch.tensor(
        [
            [[[-1, -1], [0, 2], [2, 4], [4, 6], [-1, -1]], [[-1, -1], [0, 3], [3, 6], [-1, -1], [-1, -1]]],
            [[[-1, -1], [0, 2], [2, 4], [4, 6], [-1, -1]], [[-1, -1], [0, 3], [3, 6], [-1, -1], [-1, -1]]],
        ],
        dtype=torch.long,
    )
    return {
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "candidate_char_spans": candidate_char_spans,
        "candidate_prior_probabilities": torch.tensor([[0.75, 0.25], [0.75, 0.25]]),
        "candidate_language_evidence": torch.tensor(
            [[[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]], [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]]
        ),
        "candidate_mask": torch.tensor([[True, True], [True, True]]),
        "labels": torch.tensor([[-100, 4, -100, -100, -100], [-100, 9, -100, -100, -100]]),
        "language_labels": torch.tensor([0, 0]),
    }


def _reference_alignment(
    candidate_embeddings: torch.Tensor,
    candidate_spans: torch.Tensor,
    reference_spans: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Small scalar reference for validating the vectorized alignment kernel."""

    batch_size, candidate_count, _, _ = candidate_embeddings.shape
    projected = reference_embeddings[:, None].expand(-1, candidate_count, -1, -1).clone()
    for batch_index in range(batch_size):
        for reference_index, (start, end) in enumerate(reference_spans[batch_index].tolist()):
            if start < 0 or end <= start:
                continue
            for candidate_index in range(candidate_count):
                spans = candidate_spans[batch_index, candidate_index]
                overlaps = torch.minimum(spans[:, 1], torch.tensor(end)) - torch.maximum(
                    spans[:, 0], torch.tensor(start)
                )
                valid = (spans[:, 0] >= 0) & (spans[:, 1] > spans[:, 0]) & (overlaps > 0)
                if not bool(valid.any()):
                    continue
                weights = overlaps[valid].to(candidate_embeddings.dtype)
                values = candidate_embeddings[batch_index, candidate_index, valid]
                projected[batch_index, candidate_index, reference_index] = (
                    values * weights[:, None]
                ).sum(dim=0) / weights.sum().clamp_min(1.0)
    return projected


def run_alignment_kernel_tests() -> None:
    """The chunked vectorized kernel must exactly preserve span projection."""

    torch.manual_seed(19)
    candidate_embeddings = torch.randn(2, 3, 5, 7)
    candidate_spans = torch.tensor(
        [
            [
                [[-1, -1], [0, 2], [2, 4], [4, 7], [-1, -1]],
                [[-1, -1], [0, 3], [3, 5], [5, 7], [-1, -1]],
                [[-1, -1], [0, 1], [1, 4], [4, 7], [-1, -1]],
            ],
            [
                [[-1, -1], [0, 2], [2, 4], [4, 7], [-1, -1]],
                [[-1, -1], [0, 4], [4, 7], [-1, -1], [-1, -1]],
                [[-1, -1], [0, 2], [2, 7], [-1, -1], [-1, -1]],
            ],
        ],
        dtype=torch.long,
    )
    reference_spans = candidate_spans[:, 0]
    reference_embeddings = candidate_embeddings[:, 0]
    expected = _reference_alignment(
        candidate_embeddings, candidate_spans, reference_spans, reference_embeddings
    )
    for chunk_size in (1, 2, 8):
        actual = project_candidate_embeddings_to_reference(
            candidate_embeddings,
            candidate_spans,
            reference_spans,
            reference_embeddings,
            reference_chunk_size=chunk_size,
        )
        assert torch.allclose(actual, expected), "Vectorized alignment changed character-span projection."
    sparse_reference, sparse_candidate, sparse_weights = _build_sparse_alignment_edges(candidate_spans)
    sparse_actual = project_candidate_embeddings_to_reference(
        candidate_embeddings,
        candidate_spans,
        reference_spans,
        reference_embeddings,
        sparse_reference_indices=sparse_reference,
        sparse_candidate_indices=sparse_candidate,
        sparse_overlap_weights=sparse_weights,
    )
    assert torch.allclose(sparse_actual, expected), "Sparse alignment changed character-span projection."


def run_forward_and_gradient_tests() -> None:
    model = _tiny_model()
    model.eval()
    inputs = _inputs()
    transformer_arguments: dict[str, object] = {}

    def capture_transformer_input(_module: torch.nn.Module, _args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        transformer_arguments.update(kwargs)

    hook = model.base_model.roberta.register_forward_pre_hook(capture_transformer_input, with_kwargs=True)
    enabled = model(**inputs)
    hook.remove()
    assert enabled.used_inputs_embeds is True, "Transformer must receive fused representations through inputs_embeds."
    assert transformer_arguments.get("input_ids") is None
    assert isinstance(transformer_arguments.get("inputs_embeds"), torch.Tensor)
    assert enabled.loss is not None and enabled.mlm_loss is not None

    changed_priors = {key: value.clone() for key, value in inputs.items()}
    changed_priors["candidate_prior_probabilities"] = torch.tensor([[0.10, 0.90], [0.10, 0.90]])
    with torch.no_grad():
        prior_changed = model(**changed_priors)
        no_fusion = model(**inputs, fusion_enabled=False)
    assert not torch.allclose(enabled.candidate_weights, prior_changed.candidate_weights), (
        "Changing candidate priors must change effective fusion weights."
    )
    assert not torch.allclose(enabled.logits, prior_changed.logits), (
        "Unit test failed: candidate weights did not change MLM logits."
    )
    assert not torch.allclose(enabled.logits, no_fusion.logits), (
        "Disabling fusion must change the controlled model output."
    )

    # Candidate padding must not influence the learned candidate gate.  This
    # guards against batch-dependent routing weights caused by averaging a pad
    # embedding into a shorter candidate path.
    changed_padding = {key: value.clone() for key, value in inputs.items()}
    changed_padding["candidate_input_ids"][:, 1, -1] = 14
    with torch.no_grad():
        padding_changed = model(**changed_padding)
    assert torch.allclose(enabled.candidate_weights, padding_changed.candidate_weights), (
        "Masked candidate padding must not change fusion weights."
    )

    # Force the lightweight classifier toward Spanish; candidate language
    # evidence differs, so this must alter routing-dependent fusion weights.
    with torch.no_grad():
        final_layer = model.language_classifier.network[-1]
        final_layer.bias.copy_(torch.tensor([-8.0, 8.0, -8.0]))
    with torch.no_grad():
        router_changed = model(**inputs)
    assert not torch.allclose(enabled.candidate_weights, router_changed.candidate_weights), (
        "Language-router probabilities must influence candidate fusion weights."
    )

    # MLM loss alone (without an auxiliary classifier loss) must backpropagate
    # into both the language router and the learned fusion gate.
    model.zero_grad(set_to_none=True)
    output = model(**inputs)
    assert output.mlm_loss is not None
    output.mlm_loss.backward()
    classifier_gradients = [parameter.grad for parameter in model.language_classifier.parameters() if parameter.requires_grad]
    fusion_gradients = [parameter.grad for parameter in model.fusion.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in classifier_gradients), (
        "MLM gradients did not reach every trainable language-router parameter."
    )
    assert all(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in fusion_gradients), (
        "MLM gradients did not reach every trainable fusion parameter."
    )


if __name__ == "__main__":
    run_alignment_kernel_tests()
    run_forward_and_gradient_tests()
    print("Fused MLM alignment, forward, output-dependence, and gradient-flow tests passed.")
