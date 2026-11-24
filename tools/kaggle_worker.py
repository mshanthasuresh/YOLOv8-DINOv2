"""Claim-based run queue for a time-limited Kaggle session: one worker per GPU, shared priority list.

    CUDA_VISIBLE_DEVICES=0 python -m tools.kaggle_worker --deadline <unix time> --out /kaggle/working/out NAME [NAME ...]

A worker always takes the highest-priority unclaimed run that is still expected to finish before the deadline,
and copies each finished run (minus last.pt) to --out so it survives the session.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from nafuse import DATA_ROOT, RUNS_ROOT
from nafuse.specs import experiment_matrix

# Conservative minutes on one T4 while a second run shares the CPUs; replaced by measured times once known.
EST_MINUTES = {
    ("voc", "yolov8n", False): 75,
    ("voc", "yolov8n", True): 120,
    ("voc", "yolov8s", False): 100,
    ("visdrone", "yolov8n", False): 70,
    ("visdrone", "yolov8n", True): 90,
}
VISDRONE_READY = DATA_ROOT / ".visdrone_ready"
VISDRONE_FAILED = DATA_ROOT / ".visdrone_failed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--deadline", type=float, required=True, help="unix time by which started runs must finish")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    specs = {s.name: s for s in experiment_matrix()}
    unknown = [n for n in args.names if n not in specs]
    if unknown:
        raise SystemExit(f"unknown runs: {unknown}")
    claims = RUNS_ROOT / "_claims"
    claims.mkdir(parents=True, exist_ok=True)
    measured: dict[tuple, float] = {}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GITHUB_ACTIONS": "1"}  # GITHUB_ACTIONS throttles progress bars

    while True:
        waiting = started = False
        for name in args.names:
            spec = specs[name]
            if (claims / name).exists():
                continue
            if spec.dataset == "visdrone" and not VISDRONE_READY.exists():
                waiting = waiting or not VISDRONE_FAILED.exists()
                continue
            kind = (spec.dataset, spec.model, spec.arm != "baseline")
            minutes = measured.get(kind, EST_MINUTES[kind])
            if time.time() + 60 * minutes > args.deadline:
                continue
            try:
                os.close(os.open(claims / name, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            log(f"RUN {name} (expected {minutes:.0f} min)")
            t0 = time.time()
            code = subprocess.call([sys.executable, "-m", "tools.run_one", "--name", name], env=env)
            took = (time.time() - t0) / 60
            measured[kind] = 1.1 * took
            src = RUNS_ROOT / spec.group / name
            ok = code == 0 and (src / "result.json").exists()
            log(f"{'DONE' if ok else 'FAILED'} {name} exit={code} after {took:.0f} min")
            if ok:
                dst = args.out / "runs" / spec.group / name
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("last.pt"), dirs_exist_ok=True)
            started = True
            break
        if started:
            continue
        if waiting and time.time() + 3600 < args.deadline:
            time.sleep(30)
            continue
        log("no runnable work left before the deadline")
        return


if __name__ == "__main__":
    main()
