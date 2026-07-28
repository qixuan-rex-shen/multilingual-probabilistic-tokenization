"""Check that one configured XLM-R-style training step fits the local GPU.

This diagnostic deliberately creates no checkpoints or experiment artifacts.
It is intended to catch a configuration/VRAM mismatch before a long,
checkpointed pretraining job begins.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path

import torch
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--model-kind", choices=("bpe", "fused"), default="bpe")
    parser.add_argument(
        "--real-bpe-batch",
        action="store_true",
        help="use the first scheduled BPE corpus batch instead of synthetic token IDs",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.xlmr import build_language_conditioned_fused_xlmr_mlm, build_xlmr_mlm
    from src.training.pretrain import _iter_batches, _iter_sequences, _mask_batch

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        print({"preflight": "no_cuda"})
        return 2

    device = torch.device("cuda")
    model = None
    optimizer = None
    tensors: tuple[torch.Tensor, ...] = ()
    try:
        if arguments.model_kind == "bpe":
            model = build_xlmr_mlm(config["model"], int(config["tokenizer"]["vocab_size"])).to(device)
        else:
            model = build_language_conditioned_fused_xlmr_mlm(
                config["model"],
                int(config["probabilistic_tokenizer"]["vocab_size"]),
                config["probabilistic_tokenizer"],
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        batch_size = arguments.batch_size or int(config["training"]["batch_size"])
        sequence_length = arguments.sequence_length or int(config["model"]["max_sequence_length"])
        if arguments.real_bpe_batch:
            if arguments.model_kind != "bpe":
                raise ValueError("--real-bpe-batch is only valid for --model-kind bpe.")
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(project_root / config["paths"]["tokenizers"] / "bpe"))
            records = _iter_sequences(
                project_root,
                project_root / config["paths"]["processed_data"] / "mlm_training" / "manifest.json",
                tokenizer,
                sequence_length,
                "train",
                document_sequence_policy=str(config["training"]["document_sequence_policy"]),
            )
            batch = next(_iter_batches(records, batch_size))
            masked = _mask_batch(batch, tokenizer, float(config["training"]["mask_probability"]), int(config["training"]["seed"]), device)
            input_ids = masked["input_ids"]
            attention_mask = masked["attention_mask"]
            labels = masked["labels"]
        else:
            vocabulary_size = int(config["tokenizer"]["vocab_size"])
            input_ids = torch.randint(5, vocabulary_size, (batch_size, sequence_length), device=device)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels[:, 1:] = -100
        if arguments.model_kind == "bpe":
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            tensors = (input_ids, attention_mask, labels)
        else:
            candidate_count = int(config["probabilistic_tokenizer"]["max_candidates"])
            candidate_input_ids = input_ids[:, None, :].expand(-1, candidate_count, -1).clone()
            candidate_attention_mask = attention_mask[:, None, :].expand(-1, candidate_count, -1).clone()
            candidate_spans = torch.full(
                (batch_size, candidate_count, sequence_length, 2), -1, dtype=torch.long, device=device
            )
            positions = torch.arange(sequence_length - 2, device=device)
            candidate_spans[:, :, 1:-1, 0] = positions
            candidate_spans[:, :, 1:-1, 1] = positions + 1
            model_inputs = {
                "candidate_input_ids": candidate_input_ids,
                "candidate_attention_mask": candidate_attention_mask,
                "candidate_char_spans": candidate_spans,
                "candidate_prior_probabilities": torch.full(
                    (batch_size, candidate_count), 1.0 / candidate_count, device=device
                ),
                "candidate_language_evidence": torch.full(
                    (batch_size, candidate_count, 3), 1.0 / 3.0, device=device
                ),
                "candidate_mask": torch.ones(batch_size, candidate_count, dtype=torch.bool, device=device),
                "labels": labels,
                "language_labels": torch.zeros(batch_size, dtype=torch.long, device=device),
            }
            tensors = tuple(model_inputs.values())
        torch.cuda.reset_peak_memory_stats(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(**model_inputs)
            if output.loss is None:
                raise RuntimeError("MLM preflight did not produce a loss.")
            output.loss.backward()
        optimizer.step()
        print(
            {
                "preflight": "fit",
                "model_kind": arguments.model_kind,
                "real_bpe_batch": arguments.real_bpe_batch,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            }
        )
        return 0
    except (torch.OutOfMemoryError, torch.AcceleratorError) as error:
        if "out of memory" not in str(error).lower():
            raise
        print(
            {
                "preflight": "oom",
                "model_kind": arguments.model_kind,
                "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "message": str(error).splitlines()[0],
            }
        )
        return 3
    finally:
        del tensors, model, optimizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except torch.AcceleratorError:
            # A failed CUDA allocation can poison this short-lived diagnostic
            # context.  Process exit releases it without affecting training.
            pass


if __name__ == "__main__":
    raise SystemExit(main())
