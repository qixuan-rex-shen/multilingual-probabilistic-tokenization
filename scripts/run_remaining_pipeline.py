"""Safely run fused MLM, GLueCoS fine-tuning, and analysis after the BPE worker.

This local controller is intentionally conservative: it waits for the currently
running baseline worker to write its *final* checkpoint, and it stops with a
durable failed state rather than launching another stage if that worker exits
without completing.  It never starts a second BPE job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _complete_checkpoint(path: Path) -> bool:
    return (path / "trainer_state.pt").is_file() and (path / "config.json").is_file()


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        # ``tasklist`` can return "Access denied" for a perfectly healthy GPU
        # worker launched by another local process.  PowerShell's Get-Process
        # uses the same query mechanism as the user's normal monitoring flow.
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_stage(project_root: Path, state_path: Path, state: dict[str, Any], name: str, script: str) -> None:
    state["current_stage"] = name
    state["stages"][name] = {"status": "running", "started_at_utc": _now(), "script": script}
    _atomic_json(state_path, state)
    command = [sys.executable, "-u", str(project_root / "scripts" / script)]
    print(f"Starting scheduled stage {name}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode:
        state["stages"][name].update(
            {"status": "failed", "finished_at_utc": _now(), "return_code": completed.returncode}
        )
        state["status"] = "failed"
        _atomic_json(state_path, state)
        raise RuntimeError(f"Scheduled stage {name} failed with return code {completed.returncode}.")
    state["stages"][name].update({"status": "completed", "finished_at_utc": _now()})
    _atomic_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-worker-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")

    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    state_path = project_root / Path(config["checkpointing"]["manifest_path"]).parent / "remaining_pipeline_schedule.json"
    state: dict[str, Any] = {
        "status": "waiting_for_bpe",
        "created_at_utc": _now(),
        "bpe_worker_pid": arguments.bpe_worker_pid,
        "poll_seconds": arguments.poll_seconds,
        "stages": {},
    }
    _atomic_json(state_path, state)
    checkpoint_root = project_root / config["paths"]["checkpoints"]
    bpe_final = checkpoint_root / "bpe_model" / "final"
    print(f"Waiting for BPE final checkpoint: {bpe_final}", flush=True)
    while not _complete_checkpoint(bpe_final):
        if not _pid_is_running(arguments.bpe_worker_pid):
            state["status"] = "failed"
            state["error"] = "The tracked BPE worker exited before writing a complete final checkpoint."
            state["failed_at_utc"] = _now()
            _atomic_json(state_path, state)
            raise RuntimeError(state["error"])
        time.sleep(arguments.poll_seconds)

    state["status"] = "running"
    state["bpe_completed_at_utc"] = _now()
    _atomic_json(state_path, state)
    _run_stage(project_root, state_path, state, "probabilistic_fused_mlm", "train_probabilistic_fused_mlm.py")
    probabilistic_final = checkpoint_root / "probabilistic_model" / "final"
    if not _complete_checkpoint(probabilistic_final):
        raise RuntimeError("Probabilistic MLM script succeeded without a complete final checkpoint.")
    _run_stage(project_root, state_path, state, "gluecos_finetuning", "run_gluecos_finetuning.py")
    _run_stage(project_root, state_path, state, "statistical_analysis", "run_statistical_analysis.py")
    state["current_stage"] = None
    state["status"] = "completed"
    state["completed_at_utc"] = _now()
    _atomic_json(state_path, state)
    print("All scheduled post-BPE stages completed.", flush=True)


if __name__ == "__main__":
    main()
