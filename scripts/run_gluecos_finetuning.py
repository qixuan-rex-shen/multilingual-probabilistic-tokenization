"""Run all paired local GLueCoS fine-tuning stages after MLM pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _complete_checkpoint(path: Path) -> bool:
    return (path / "trainer_state.pt").is_file() and (path / "config.json").is_file()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.training.finetune import run_gluecos_finetuning

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    checkpoint_root = project_root / config["paths"]["checkpoints"]
    bpe = checkpoint_root / "bpe_model" / "best"
    probabilistic = checkpoint_root / "probabilistic_model" / "best"
    for checkpoint in (bpe, probabilistic):
        if not _complete_checkpoint(checkpoint):
            raise FileNotFoundError(f"A complete selected MLM checkpoint is required: {checkpoint}")
    print(run_gluecos_finetuning(project_root, config, bpe, probabilistic))


if __name__ == "__main__":
    main()
