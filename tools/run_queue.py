"""Run the whole experiment matrix sequentially, one fresh subprocess per job (resumable).

    python -m tools.run_queue                  # everything, priority order
    python -m tools.run_queue --max-priority 1 # headline + ablation only
    python -m tools.run_queue --dry-run

Finished runs (runs/<group>/<name>/result.json) are skipped. If the machine shuts down mid-run, restart
the queue after boot; that run resumes from weights/last.pt (model, optimizer, EMA and epoch), while finished
runs stay skipped. If training finished but evaluation did not, it proceeds directly to evaluation.
Create the file runs/STOP to stop cleanly after the current job. After every finished run the quick analysis
and slide deck are regenerated; the full analysis (with bootstrap CIs) runs when the queue empties.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from nafuse import DATA_ROOT, PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT
from nafuse.runner import run_dir
from nafuse.specs import experiment_matrix

STATUS = RUNS_ROOT / "queue_status.json"
STOP = RUNS_ROOT / "STOP"
LOGS = RUNS_ROOT / "logs"


def keep_awake() -> None:
    """Block idle sleep while this process runs (Windows only; released automatically on exit)."""
    if os.name == "nt":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def _module_exists(name: str) -> bool:
    return (PROJECT_ROOT / Path(*name.split("."))).with_suffix(".py").exists()


def plan_jobs(max_priority: int) -> list[tuple[str, list[str]]]:
    py = sys.executable
    jobs: list[tuple[str, list[str]]] = []
    if not (RESULTS_ROOT / "bench.json").exists():
        jobs.append(("bench", [py, "-m", "tools.bench"]))
    if not (RESULTS_ROOT / "token_stats.json").exists() and _module_exists("tools.token_stats"):
        jobs.append(("token_stats", [py, "-m", "tools.token_stats"]))
    visdrone_queued = False
    for spec in experiment_matrix():
        if spec.priority > max_priority or (run_dir(spec) / "result.json").exists():
            continue
        if spec.dataset == "visdrone" and not visdrone_queued and not (DATA_ROOT / "visdrone.yaml").exists():
            jobs.append(("prepare_visdrone", [py, "-m", "tools.prepare_data", "--visdrone"]))
            visdrone_queued = True
        jobs.append((spec.name, [py, "-m", "tools.run_one", "--name", spec.name]))
    return jobs


def _run(name: str, cmd: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GITHUB_ACTIONS": "1"}  # GITHUB_ACTIONS only throttles Ultralytics progress bars
    with open(LOGS / f"{name}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def _write_status(**fields) -> None:
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    status.update(fields, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    STATUS.write_text(json.dumps(status, indent=2))


def refresh_report(full: bool) -> None:
    py = sys.executable
    if _module_exists("tools.analyze"):
        _run("analyze", [py, "-m", "tools.analyze"] + ([] if full else ["--quick"]))
    if _module_exists("tools.make_slides"):
        _run("make_slides", [py, "-m", "tools.make_slides"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-priority", type=int, default=99)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = plan_jobs(args.max_priority)
    if args.dry_run:
        for name, _ in jobs:
            print(name)
        print(f"{len(jobs)} jobs")
        return

    keep_awake()
    STOP.unlink(missing_ok=True)
    _write_status(pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S"), done=[], failed=[], current=None)
    done, failed = [], []
    for i, (name, cmd) in enumerate(jobs):
        if STOP.exists():
            _write_status(current=None, stopped=True)
            return
        _write_status(current=name, remaining=len(jobs) - i)
        start = time.time()
        rc = _run(name, cmd)
        if rc != 0:  # one retry covers transient CUDA/dataloader failures
            rc = _run(name, cmd)
        (done if rc == 0 else failed).append({"job": name, "minutes": round((time.time() - start) / 60, 1), "rc": rc})
        _write_status(done=done, failed=failed)
        if rc == 0 and name not in {"bench", "token_stats", "prepare_visdrone"}:
            refresh_report(full=False)
    _write_status(current="final analysis")
    refresh_report(full=True)
    _write_status(current=None, finished=time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
