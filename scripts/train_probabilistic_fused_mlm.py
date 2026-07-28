"""Run the checkpointed language-conditioned fused MLM stage from the project root."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import yaml


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.training.pretrain import train_language_conditioned_fused_mlm

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    result = train_language_conditioned_fused_mlm(
        project_root=project_root,
        plan_path=project_root / config["paths"]["processed_data"] / "mlm_training" / "manifest.json",
        tokenizer_directory=project_root / config["paths"]["tokenizers"] / "probabilistic",
        language_classifier_directory=project_root / config["paths"]["language_classifier"],
        model_settings=config["model"],
        probabilistic_settings=config["probabilistic_tokenizer"],
        training=config["training"],
        destination=project_root / config["paths"]["checkpoints"] / "probabilistic_model",
        runtime_mode=config["runtime"]["mode"],
        logs_directory=project_root / config["paths"]["logs"],
        results_directory=project_root / config["paths"]["results"],
    )
    print(result)


if __name__ == "__main__":
    main()
