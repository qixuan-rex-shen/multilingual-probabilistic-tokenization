"""Generate paired GLueCoS statistical reports from saved fine-tuning results."""

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
    from src.evaluation.analysis import analyze_finetuning_results, write_experiment_summary

    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    report = analyze_finetuning_results(project_root, config)
    summary = write_experiment_summary(project_root, config, report)
    print({"report": report, "summary": summary})


if __name__ == "__main__":
    main()
