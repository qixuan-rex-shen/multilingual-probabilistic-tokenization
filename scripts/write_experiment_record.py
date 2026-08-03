"""Overwrite the tracked consolidated record for the configured experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.evaluation.experiment_record import write_experiment_record

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    print(write_experiment_record(project_root, config))


if __name__ == "__main__":
    main()
