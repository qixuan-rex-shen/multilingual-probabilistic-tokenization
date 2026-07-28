"""Bounded performance and correctness smoke test for the fused MLM path.

The script uses the real frozen probabilistic tokenizer, language prior, corpus
schedule, model configuration, masking rule, and a single training microbatch.
It deliberately writes metrics only, never a model checkpoint.
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--microbatches",
        type=int,
        default=1,
        help="Number of real microbatches to benchmark without writing a checkpoint.",
    )
    arguments = parser.parse_args()
    if arguments.microbatches <= 0:
        raise ValueError("--microbatches must be positive.")
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.language.language_classifier import CulturaXLanguagePriorClassifier
    from src.models.xlmr import build_language_conditioned_fused_xlmr_mlm
    from src.tokenizer.probabilistic import UnigramCandidateTokenizer
    from src.training.pretrain import (
        _iter_candidate_batches,
        _iter_candidate_examples,
        _iter_parallel_candidate_batches,
        _mask_candidate_batch,
        _prefetch_iterator,
    )

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    training = config["training"]
    model_settings = config["model"]
    tokenizer = UnigramCandidateTokenizer.from_pretrained(
        project_root / config["paths"]["tokenizers"] / "probabilistic"
    )
    candidate_selection = tokenizer.configure_candidate_selection(config["probabilistic_tokenizer"])
    language_prior = CulturaXLanguagePriorClassifier.load(
        project_root / config["paths"]["language_classifier"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and training["precision"] == "fp16_or_bfloat16_if_available"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    plan_path = project_root / config["paths"]["processed_data"] / "mlm_training" / "manifest.json"
    candidate_worker_count = int(training.get("candidate_preparation_workers", 0))
    candidate_buffer_batches = int(training.get("candidate_preparation_buffer_batches", 1))
    serial_reference_batch = None
    if candidate_worker_count:
        # The first parallel batch must be byte-for-byte equivalent in logical
        # content to the serial path before this smoke test reports success.
        serial_reference_batch = next(
            _iter_candidate_batches(
                _iter_candidate_examples(
                    project_root,
                    plan_path,
                    tokenizer,
                    language_prior,
                    int(model_settings["max_sequence_length"]),
                    "train",
                    document_sequence_policy=str(training["document_sequence_policy"]),
                ),
                int(training["batch_size"]),
            )
        )
    if candidate_worker_count:
        candidate_batches = _iter_parallel_candidate_batches(
            project_root,
            plan_path,
            project_root / config["paths"]["tokenizers"] / "probabilistic",
            project_root / config["paths"]["language_classifier"],
            config["probabilistic_tokenizer"],
            int(model_settings["max_sequence_length"]),
            "train",
            int(training["batch_size"]),
            str(training["document_sequence_policy"]),
            candidate_worker_count,
            candidate_buffer_batches,
        )
    else:
        candidate_batches = _iter_candidate_batches(
            _iter_candidate_examples(
                project_root,
                plan_path,
                tokenizer,
                language_prior,
                int(model_settings["max_sequence_length"]),
                "train",
                document_sequence_policy=str(training["document_sequence_policy"]),
            ),
            int(training["batch_size"]),
        )
    torch.manual_seed(int(training["seed"]))
    model = build_language_conditioned_fused_xlmr_mlm(
        model_settings, len(tokenizer.tokenizer), config["probabilistic_tokenizer"]
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prefetched_batches = (
        candidate_batches
        if candidate_worker_count
        else _prefetch_iterator(candidate_batches, int(training.get("candidate_prefetch_batches", 0)))
    )
    candidate_wait_seconds = 0.0
    masking_seconds = 0.0
    model_seconds = 0.0
    candidate_wait_seconds_by_microbatch: list[float] = []
    forward_backward_seconds_by_microbatch: list[float] = []
    total_started = time.perf_counter()
    output = None
    gradient_norm = 0.0
    inputs: dict[str, torch.Tensor] | None = None
    observed_candidate_counts: list[int] = []
    candidate_equivalence_verified = serial_reference_batch is None

    def candidate_signature(batch: list[tuple[list[object], int]]) -> list[tuple[int, list[tuple[object, ...]]]]:
        return [
            (
                language_id,
                [
                    (
                        tuple(candidate.input_ids),
                        tuple(candidate.offsets),
                        candidate.token_score,
                        candidate.prior_probability,
                        tuple(candidate.language_evidence),
                    )
                    for candidate in candidates
                ],
            )
            for candidates, language_id in batch
        ]

    try:
        for microbatch_index in range(arguments.microbatches):
            started = time.perf_counter()
            batch = next(prefetched_batches)
            if microbatch_index == 0 and serial_reference_batch is not None:
                if candidate_signature(batch) != candidate_signature(serial_reference_batch):
                    raise AssertionError("Parallel candidate preparation changed the scheduled candidate batch.")
                candidate_equivalence_verified = True
            observed_candidate_counts.extend(len(candidates) for candidates, _ in batch)
            candidate_wait = time.perf_counter() - started
            candidate_wait_seconds += candidate_wait
            candidate_wait_seconds_by_microbatch.append(candidate_wait)
            started = time.perf_counter()
            inputs = _mask_candidate_batch(
                batch,
                tokenizer,
                float(training["mask_probability"]),
                int(training["seed"]) + microbatch_index // int(training["gradient_accumulation_steps"]),
                device,
            )
            _sync(device)
            masking_seconds += time.perf_counter() - started
            started = time.perf_counter()
            autocast = torch.amp.autocast(device_type="cuda", enabled=use_fp16) if device.type == "cuda" else nullcontext()
            with autocast:
                output = model(**inputs)
                if output.loss is None or output.mlm_loss is None:
                    raise RuntimeError("Fused smoke test did not produce an MLM loss.")
                loss = output.loss / int(training["gradient_accumulation_steps"])
            scaler.scale(loss).backward() if use_fp16 else loss.backward()
            if (microbatch_index + 1) % int(training["gradient_accumulation_steps"]) == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clipping"]))
                )
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            _sync(device)
            model_elapsed = time.perf_counter() - started
            model_seconds += model_elapsed
            forward_backward_seconds_by_microbatch.append(model_elapsed)
    finally:
        prefetched_batches.close()
    total_seconds = time.perf_counter() - total_started
    if output is None or inputs is None:
        raise RuntimeError("Smoke test did not process a microbatch.")

    result = {
        "device": str(device),
        "batch_size": int(training["batch_size"]),
        "microbatches": arguments.microbatches,
        "candidate_prefetch_batches": int(training.get("candidate_prefetch_batches", 0)),
        "candidate_preparation_workers": candidate_worker_count,
        "candidate_preparation_buffer_batches": candidate_buffer_batches,
        "candidate_equivalence_verified": candidate_equivalence_verified,
        "candidate_selection": candidate_selection,
        "observed_candidate_counts": observed_candidate_counts,
        "max_candidate_length": int(inputs["candidate_input_ids"].shape[-1]),
        "candidate_wait_seconds": candidate_wait_seconds,
        "candidate_wait_seconds_by_microbatch": candidate_wait_seconds_by_microbatch,
        "alignment_masking_seconds": masking_seconds,
        "forward_backward_seconds": model_seconds,
        "forward_backward_seconds_by_microbatch": forward_backward_seconds_by_microbatch,
        "total_wall_seconds": total_seconds,
        "estimated_optimizer_step_seconds": total_seconds
        * int(training["gradient_accumulation_steps"]) / arguments.microbatches,
        "mlm_loss": float(output.mlm_loss.detach().cpu()),
        "total_loss": float(output.loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "used_inputs_embeds": bool(output.used_inputs_embeds),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    result_path = project_root / config["paths"]["results"] / "mlm" / "probabilistic_fused_smoke.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved smoke metrics to: {result_path}")


if __name__ == "__main__":
    main()
