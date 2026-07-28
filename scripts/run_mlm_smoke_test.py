"""Exercise one complete tiny pretraining step for each paired MLM arm.

The fixture streams real local CulturaX records and uses the frozen production
tokenizers.  It runs one BPE and one probabilistic-fusion optimizer step on
CPU, including validation, checkpoint selection, final checkpoint, and
held-out test evaluation.  It never reads or writes the live experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _complete(path: Path, fused: bool) -> bool:
    required = [path / "config.json", path / "trainer_state.pt"]
    if fused:
        required += [path / "candidate_fusion_state.pt", path / "candidate_fusion_config.json"]
    return all(item.is_file() for item in required)


def _fixture_plan(project_root: Path, source_plan: Path, fixture_root: Path) -> Path:
    """Make the immutable corpus plan portable to the isolated fixture root."""

    plan = json.loads(source_plan.read_text(encoding="utf-8"))
    for language in plan["languages"]:
        language["source_directory"] = str((project_root / language["source_directory"]).resolve())
        language["part_paths"] = [str((project_root / part).resolve()) for part in language["part_paths"]]
    destination = fixture_root / "mlm_training_plan.json"
    print(f"Saving checkpoint to: {destination}")
    _atomic_json(destination, plan)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="outputs/test_fixtures/mlm_smoke")
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.training.pretrain import train_from_scratch_mlm, train_language_conditioned_fused_mlm

    # The full controller will use CUDA. This isolated check deliberately runs
    # CPU-only so it cannot compete with the active long-running BPE process.
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"-1", ""}:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 before running this CPU-only smoke test.")

    fixture_root = (project_root / arguments.output_directory).resolve()
    if project_root not in fixture_root.parents:
        raise ValueError("--output-directory must stay inside PROJECT_ROOT.")
    if fixture_root.exists():
        raise FileExistsError(f"Smoke fixture directory already exists: {fixture_root}")
    fixture_root.mkdir(parents=True)
    live_config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(live_config)
    config["project"]["root"] = "."
    config["experiment"]["id"] = "mlm_smoke"
    config["paths"]["experiments"] = "experiments"
    config_path = fixture_root / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {config_path}")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan = _fixture_plan(
        project_root,
        project_root / live_config["paths"]["processed_data"] / "mlm_training" / "manifest.json",
        fixture_root,
    )
    tiny_model = {
        **live_config["model"],
        "profile": "smoke_only",
        "layers": 1,
        "hidden_size": 32,
        "attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 66,
        "max_sequence_length": 64,
    }
    tiny_training = {
        **live_config["training"],
        "batch_size": 1,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "maximum_optimizer_steps": 1,
        "evaluation_interval": 1,
        "checkpoint_interval": 1,
        "maximum_retained_checkpoints": 1,
        "early_stopping_patience": 2,
        "validation_batches_per_language": 1,
        "precision": "fp32",
        "warmup_ratio": 0.0,
    }
    logs = fixture_root / "logs"
    results = fixture_root / "results"
    bpe_destination = fixture_root / "checkpoints" / "bpe_model"
    fused_destination = fixture_root / "checkpoints" / "probabilistic_model"
    bpe = train_from_scratch_mlm(
        fixture_root,
        plan,
        project_root / live_config["paths"]["tokenizers"] / "bpe",
        tiny_model,
        tiny_training,
        bpe_destination,
        "bpe_model",
        "train",
        logs,
        results,
    )
    fused = train_language_conditioned_fused_mlm(
        fixture_root,
        plan,
        project_root / live_config["paths"]["tokenizers"] / "probabilistic",
        project_root / live_config["paths"]["language_classifier"],
        tiny_model,
        live_config["probabilistic_tokenizer"],
        tiny_training,
        fused_destination,
        "train",
        logs,
        results,
    )
    # A completed one-step fixture has no remaining updates, but resume still
    # restores the exact saved model/optimizer/RNG state, resolves the latest
    # checkpoint, and performs the configured final checkpoint/test workflow.
    resumed_bpe = train_from_scratch_mlm(
        fixture_root,
        plan,
        project_root / live_config["paths"]["tokenizers"] / "bpe",
        tiny_model,
        tiny_training,
        bpe_destination,
        "bpe_model",
        "resume",
        logs,
        results,
    )
    resumed_fused = train_language_conditioned_fused_mlm(
        fixture_root,
        plan,
        project_root / live_config["paths"]["tokenizers"] / "probabilistic",
        project_root / live_config["paths"]["language_classifier"],
        tiny_model,
        live_config["probabilistic_tokenizer"],
        tiny_training,
        fused_destination,
        "resume",
        logs,
        results,
    )
    if not _complete(bpe_destination / "best", False) or not _complete(bpe_destination / "final", False):
        raise AssertionError("BPE pretraining smoke test did not produce complete selected and final checkpoints.")
    if not _complete(fused_destination / "best", True) or not _complete(fused_destination / "final", True):
        raise AssertionError("Fused pretraining smoke test did not produce complete selected and final checkpoints.")
    summary = fixture_root / "smoke_summary.json"
    print(f"Saving checkpoint to: {summary}")
    _atomic_json(
        summary,
        {
            "status": "passed",
            "bpe": bpe,
            "probabilistic": fused,
            "resume_bpe": resumed_bpe,
            "resume_probabilistic": resumed_fused,
        },
    )
    print("Paired MLM smoke test passed.")


if __name__ == "__main__":
    main()
