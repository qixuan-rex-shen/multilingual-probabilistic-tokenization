"""Character-span alignment primitives for candidate tokenization paths."""

from __future__ import annotations

import torch


def project_candidate_embeddings_to_reference(
    candidate_embeddings: torch.Tensor,
    candidate_spans: torch.Tensor,
    reference_spans: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_chunk_size: int = 64,
    sparse_reference_indices: torch.Tensor | None = None,
    sparse_candidate_indices: torch.Tensor | None = None,
    sparse_overlap_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project every candidate path onto reference positions by character overlap.

    Args:
        candidate_embeddings: ``[batch, candidates, candidate_tokens, hidden]``.
        candidate_spans: original-text offsets ``[batch, candidates, candidate_tokens, 2]``.
        reference_spans: original-text offsets ``[batch, reference_tokens, 2]``.
        reference_embeddings: fallback embeddings ``[batch, reference_tokens, hidden]``.

    Special/padding tokens use ``(-1, -1)`` spans and retain the reference
    embedding.  This deliberately never treats equally numbered token indices
    as aligned.
    """

    if candidate_embeddings.ndim != 4 or candidate_spans.ndim != 4 or reference_spans.ndim != 3:
        raise ValueError("Invalid candidate/reference alignment tensor ranks.")
    batch_size, candidates, _, hidden_size = candidate_embeddings.shape
    if candidate_spans.shape[:3] != candidate_embeddings.shape[:3] or candidate_spans.shape[-1] != 2:
        raise ValueError("candidate_spans must match candidate embeddings and end in a span pair.")
    if reference_spans.shape != (batch_size, reference_embeddings.shape[1], 2):
        raise ValueError("reference_spans must match reference embeddings.")

    sparse_arguments = (sparse_reference_indices, sparse_candidate_indices, sparse_overlap_weights)
    if any(argument is not None for argument in sparse_arguments) and not all(
        argument is not None for argument in sparse_arguments
    ):
        raise ValueError("All sparse alignment tensors must be supplied together.")
    if all(argument is not None for argument in sparse_arguments):
        assert sparse_reference_indices is not None
        assert sparse_candidate_indices is not None
        assert sparse_overlap_weights is not None
        if sparse_reference_indices.ndim != 3:
            raise ValueError("sparse alignment indices must be [batch, candidates, edges].")
        expected_edges = sparse_reference_indices.shape
        if sparse_candidate_indices.shape != expected_edges or sparse_overlap_weights.shape != expected_edges:
            raise ValueError("Sparse alignment tensors must share a [batch, candidates, edges] shape.")
        if expected_edges[:2] != (batch_size, candidates):
            raise ValueError("Sparse alignment tensors must match batch and candidate dimensions.")
        if sparse_reference_indices.numel() and (
            sparse_reference_indices.min() < 0 or sparse_reference_indices.max() >= reference_embeddings.shape[1]
        ):
            raise ValueError("Sparse reference indices are outside the reference sequence.")
        if sparse_candidate_indices.numel() and (
            sparse_candidate_indices.min() < 0 or sparse_candidate_indices.max() >= candidate_embeddings.shape[2]
        ):
            raise ValueError("Sparse candidate indices are outside the candidate sequence.")

        edge_count = expected_edges[-1]
        if edge_count == 0:
            return reference_embeddings[:, None].expand(-1, candidates, -1, -1)
        gathered_embeddings = torch.gather(
            candidate_embeddings,
            2,
            sparse_candidate_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size),
        )
        weights = sparse_overlap_weights.to(candidate_embeddings.dtype)
        weighted_embeddings = gathered_embeddings * weights.unsqueeze(-1)
        projected_sums = torch.zeros(
            batch_size,
            candidates,
            reference_embeddings.shape[1],
            hidden_size,
            device=candidate_embeddings.device,
            dtype=candidate_embeddings.dtype,
        )
        projected_sums.scatter_add_(
            2,
            sparse_reference_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size),
            weighted_embeddings,
        )
        weight_sums = torch.zeros(
            batch_size,
            candidates,
            reference_embeddings.shape[1],
            device=candidate_embeddings.device,
            dtype=candidate_embeddings.dtype,
        )
        weight_sums.scatter_add_(2, sparse_reference_indices, weights)
        projected = projected_sums / weight_sums.clamp_min(1.0).unsqueeze(-1)
        fallback = reference_embeddings[:, None].expand(-1, candidates, -1, -1)
        return torch.where(weight_sums.unsqueeze(-1) > 0, projected, fallback)

    if reference_chunk_size <= 0:
        raise ValueError("reference_chunk_size must be positive.")

    # The original implementation performed a Python loop for every
    # (batch, reference-token, candidate) triple.  For a 8 x 5 x 512 batch
    # that resulted in 20,480 tiny CPU/GPU dispatches before every transformer
    # call.  The operation below is mathematically identical: character-span
    # overlap is used as the aggregation weight, and positions with no valid
    # overlap retain their reference embedding.  Chunking reference positions
    # bounds the temporary [B, C, R_chunk, L_candidate] tensor on an 8 GB GPU.
    candidate_start = candidate_spans[..., 0]
    candidate_end = candidate_spans[..., 1]
    valid_candidate = (candidate_start >= 0) & (candidate_end > candidate_start)
    projected_chunks: list[torch.Tensor] = []
    reference_length = reference_embeddings.shape[1]
    for chunk_start in range(0, reference_length, reference_chunk_size):
        chunk_end = min(reference_length, chunk_start + reference_chunk_size)
        reference_chunk = reference_spans[:, chunk_start:chunk_end]
        reference_start = reference_chunk[..., 0]
        reference_end = reference_chunk[..., 1]
        valid_reference = (reference_start >= 0) & (reference_end > reference_start)

        overlaps = (
            torch.minimum(candidate_end[:, :, None, :], reference_end[:, None, :, None])
            - torch.maximum(candidate_start[:, :, None, :], reference_start[:, None, :, None])
        ).clamp_min(0)
        overlap_weights = overlaps.to(candidate_embeddings.dtype)
        overlap_weights = overlap_weights * valid_candidate[:, :, None, :].to(candidate_embeddings.dtype)
        overlap_weights = overlap_weights * valid_reference[:, None, :, None].to(candidate_embeddings.dtype)
        weight_sums = overlap_weights.sum(dim=-1)
        projected_chunk = torch.einsum("bcrl,bclh->bcrh", overlap_weights, candidate_embeddings)
        projected_chunk = projected_chunk / weight_sums.clamp_min(1.0).unsqueeze(-1)

        fallback = reference_embeddings[:, None, chunk_start:chunk_end].expand(-1, candidates, -1, -1)
        projected_chunks.append(torch.where(weight_sums.unsqueeze(-1) > 0, projected_chunk, fallback))
    return torch.cat(projected_chunks, dim=2)
