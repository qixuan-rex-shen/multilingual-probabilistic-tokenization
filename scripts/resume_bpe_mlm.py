"""Resume the checkpointed BPE baseline MLM stage from its latest checkpoint.

This entry point intentionally forces ``runtime_mode='resume'`` so a power
interruption cannot accidentally start a second baseline model from scratch.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    logs_directory = project_root / "outputs" / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_directory / "bpe_mlm_resume.stdout.log"
    stderr_path = logs_directory / "bpe_mlm_resume.stderr.log"
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "a", encoding="utf-8", buffering=1
    ) as stderr, redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            from src.training.pretrain import train_from_scratch_mlm

            config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
            destination = project_root / config["paths"]["checkpoints"] / "bpe_model"
            print(f"Resuming BPE MLM from the latest checkpoint in: {destination}", flush=True)
            result = train_from_scratch_mlm(
                project_root=project_root,
                plan_path=project_root / config["paths"]["processed_data"] / "mlm_training" / "manifest.json",
                tokenizer_directory=project_root / config["paths"]["tokenizers"] / "bpe",
                model_settings=config["model"],
                training=config["training"],
                destination=destination,
                group_name="bpe_model",
                runtime_mode="resume",
                logs_directory=project_root / config["paths"]["logs"],
                results_directory=project_root / config["paths"]["results"],
            )
            print(result, flush=True)
        except Exception:
            traceback.print_exc(file=stderr)
            raise


if __name__ == "__main__":
    main()
