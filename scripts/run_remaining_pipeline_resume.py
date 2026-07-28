"""Launch the post-BPE pipeline controller with durable console logs."""

from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-worker-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=300, type=int)
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    logs_directory = project_root / "outputs" / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_directory / "remaining_pipeline_resume.stdout.log"
    stderr_path = logs_directory / "remaining_pipeline_resume.stderr.log"

    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "a", encoding="utf-8", buffering=1
    ) as stderr, redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            if str(project_root / "scripts") not in sys.path:
                sys.path.insert(0, str(project_root / "scripts"))
            from run_remaining_pipeline import main as controller_main

            sys.argv = [
                "run_remaining_pipeline.py",
                "--bpe-worker-pid",
                str(arguments.bpe_worker_pid),
                "--poll-seconds",
                str(arguments.poll_seconds),
            ]
            controller_main()
        except Exception:
            traceback.print_exc(file=stderr)
            raise


if __name__ == "__main__":
    main()
