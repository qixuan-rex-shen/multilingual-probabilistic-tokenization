"""Run a tiny, local paired GLueCoS smoke test without touching real outputs.

It uses the frozen production tokenizers and language prior, but creates two
one-layer XLM-R-style MLM fixtures with the same vocabulary size.  A bounded
NER subset then exercises the actual BPE and fused candidate paths, selection
checkpoint writes, result serialization, and paired statistical report.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import PreTrainedTokenizerFast


def _complete_model(path: Path, fused: bool) -> bool:
    base = (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    )
    return base and (not fused or (path / "candidate_fusion_state.pt").is_file())


def _save_fixture_models(project_root: Path, config: dict, destination: Path) -> tuple[Path, Path]:
    from src.models.xlmr import build_language_conditioned_fused_xlmr_mlm, build_xlmr_mlm
    from src.tokenizer.probabilistic import UnigramCandidateTokenizer

    bpe = PreTrainedTokenizerFast.from_pretrained(str(project_root / config["paths"]["tokenizers"] / "bpe"))
    probabilistic = UnigramCandidateTokenizer.from_pretrained(
        project_root / config["paths"]["tokenizers"] / "probabilistic"
    )
    bpe_vocab = int(bpe.vocab_size)
    probabilistic_vocab = int(probabilistic.tokenizer.vocab_size)
    if bpe_vocab != probabilistic_vocab:
        raise ValueError(f"Smoke test requires matched tokenizer sizes, got {bpe_vocab} and {probabilistic_vocab}.")
    tiny_model = {
        **config["model"],
        "profile": "smoke_only",
        "layers": 1,
        "hidden_size": 32,
        "attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 66,
        "max_sequence_length": 64,
    }
    bpe_destination = destination / "mlm" / "bpe" / "best"
    fused_destination = destination / "mlm" / "probabilistic" / "best"
    for path, fused in ((bpe_destination, False), (fused_destination, True)):
        if path.exists() and not _complete_model(path, fused):
            raise FileExistsError(f"Incomplete smoke fixture exists and will not be overwritten: {path}")
    if not _complete_model(bpe_destination, False):
        print(f"Saving checkpoint to: {bpe_destination}")
        build_xlmr_mlm(tiny_model, bpe_vocab).save_pretrained(str(bpe_destination))
        torch.save({"smoke_fixture": True}, bpe_destination / "trainer_state.pt")
    if not _complete_model(fused_destination, True):
        print(f"Saving checkpoint to: {fused_destination}")
        build_language_conditioned_fused_xlmr_mlm(
            tiny_model, probabilistic_vocab, config["probabilistic_tokenizer"]
        ).save_pretrained(fused_destination)
        torch.save({"smoke_fixture": True}, fused_destination / "trainer_state.pt")
    return bpe_destination, fused_destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="outputs/test_fixtures/downstream_smoke",
        help="A project-relative empty directory for this disposable-but-preserved fixture.",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.evaluation.analysis import analyze_finetuning_results
    from src.training.finetune import run_gluecos_finetuning

    output_directory = (project_root / arguments.output_directory).resolve()
    if project_root not in output_directory.parents:
        raise ValueError("--output-directory must stay within PROJECT_ROOT.")
    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    bpe_checkpoint, fused_checkpoint = _save_fixture_models(project_root, config, output_directory)
    config["runtime"]["mode"] = "train"
    config["experiment"]["id"] = "downstream_smoke"
    config["model"]["max_sequence_length"] = 64
    config["data_download"]["gluecos"] = ["Huggmachas/GLuecos_NER_EN_HI"]
    config["evaluation"]["seeds"] = [1]
    config["evaluation"]["paired_bootstrap_samples"] = 25
    config["paths"]["results"] = str(output_directory.relative_to(project_root) / "results")
    config["paths"]["checkpoints"] = str(output_directory.relative_to(project_root) / "checkpoints")
    config["paths"]["downstream_checkpoints"] = str(
        output_directory.relative_to(project_root) / "checkpoints" / "downstream"
    )
    config["paths"]["logs"] = str(output_directory.relative_to(project_root) / "logs")
    config["paths"]["experiments"] = str(output_directory.relative_to(project_root) / "experiments")
    config["finetuning"].update({"batch_size": 2, "epochs": 1, "max_examples_per_split": 4})
    result = run_gluecos_finetuning(project_root, config, bpe_checkpoint, fused_checkpoint)
    report = analyze_finetuning_results(project_root, config, result)
    seed = result["Huggmachas/GLuecos_NER_EN_HI"]["seeds"]["1"]
    if seed["bpe"]["labels"] != seed["probabilistic"]["labels"]:
        raise AssertionError("Smoke test found unaligned paired downstream references.")
    if not report["tasks"]:
        raise AssertionError("Smoke test did not produce a paired statistical report.")
    summary = output_directory / "smoke_summary.json"
    print(f"Saving checkpoint to: {summary}")
    summary.write_text(
        json.dumps(
            {
                "status": "passed",
                "bpe_checkpoint": str(bpe_checkpoint),
                "probabilistic_checkpoint": str(fused_checkpoint),
                "bpe_metrics": seed["bpe"]["metrics"],
                "probabilistic_metrics": seed["probabilistic"]["metrics"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Paired downstream smoke test passed.")


if __name__ == "__main__":
    main()
