"""Print queue progress, the current run's epoch, finished results and a rough ETA.

    python -m tools.status
"""

from __future__ import annotations

import json
import time

import pandas as pd

from nafuse import RUNS_ROOT
from nafuse.runner import run_dir
from nafuse.specs import experiment_matrix
from nafuse.stats import load_results


def main() -> None:
    status_path = RUNS_ROOT / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    print(f"queue pid {status.get('pid')}  started {status.get('started')}  current: {status.get('current')}")
    for failure in status.get("failed", []):
        print(f"  FAILED: {failure['job']} (rc {failure['rc']}) -> see runs/logs/{failure['job']}.log")

    specs = experiment_matrix()
    current = next((s for s in specs if s.name == status.get("current")), None)
    if current is not None:
        csv = run_dir(current) / "results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df.columns = [c.strip() for c in df.columns]
            last = df.iloc[-1]
            print(
                f"  {current.name}: epoch {int(last['epoch'])}/{current.epochs}, "
                f"dev mAP50-95 {100 * last['metrics/mAP50-95(B)']:.1f}, {last['time'] / 60:.0f} min elapsed"
            )

    done = [s for s in specs if (run_dir(s) / "result.json").exists()]
    todo = [s for s in specs if s not in done]
    minutes = [j["minutes"] for j in status.get("done", []) if j["job"] in {s.name for s in specs}]
    if minutes:
        eta_h = sum(minutes) / len(minutes) * len(todo) / 60
        finish = time.strftime("%a %d %b %H:%M", time.localtime(time.time() + eta_h * 3600))
        print(f"{len(done)}/{len(specs)} runs finished; ~{eta_h:.0f} h of queue left (rough ETA {finish})")
    else:
        print(f"{len(done)}/{len(specs)} runs finished")

    results = load_results()
    if not results.empty:
        results = results[results["group"] != "smoke"].sort_values(["group", "dataset", "arm", "dino_res", "seed"])
        print(f"\n{'run':34s} {'test mAP50-95':>13s} {'mAP50':>7s} {'dev':>6s} {'hours':>6s}")
        for r in results.itertuples():
            print(f"{r.name:34s} {100 * r.test_map:13.2f} {100 * r.test_map50:7.2f} {100 * r.dev_map:6.2f} {r.train_hours:6.2f}")


if __name__ == "__main__":
    main()
