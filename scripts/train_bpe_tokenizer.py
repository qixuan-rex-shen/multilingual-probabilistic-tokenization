"""Run the restart-safe baseline BPE tokenizer fit from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.tokenizer.bpe import train_bpe_tokenizer

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    result = train_bpe_tokenizer(
        project_root=project_root,
        plan_path=project_root / config["paths"]["processed_data"] / "tokenizer_training" / "manifest.json",
        tokenizer_config=config["tokenizer"],
        max_sequence_length=int(config["model"]["max_sequence_length"]),
        destination=project_root / config["paths"]["tokenizers"] / "bpe",
    )
    print(result)


if __name__ == "__main__":
    main()
