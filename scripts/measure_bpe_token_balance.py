"""Measure final language token shares for the frozen baseline BPE artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.tokenizer.balance import measure_bpe_token_balance

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    result = measure_bpe_token_balance(
        project_root=project_root,
        plan_path=project_root / config["paths"]["processed_data"] / "tokenizer_training" / "manifest.json",
        tokenizer_config=config["tokenizer"],
        tokenizer_directory=project_root / config["paths"]["tokenizers"] / "bpe",
        destination=project_root / config["paths"]["results"] / "tokenizer_balance" / "bpe_token_balance.json",
    )
    print(result)


if __name__ == "__main__":
    main()
