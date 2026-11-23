"""Train + evaluate a single run from the matrix (used by the queue as a fresh subprocess).

    python -m tools.run_one --name voc-yolov8n-lngn-s0
    python -m tools.run_one --smoke --arm lngn       # 1 epoch on 2% of VOC, written to runs/smoke/
"""

from __future__ import annotations

import argparse
import json

from nafuse.runner import train_and_evaluate
from nafuse.specs import RunSpec, experiment_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", default="lngn")
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        spec = RunSpec(args.arm, 0, group="smoke", epochs=args.epochs, train={"fraction": args.fraction})
    else:
        matches = [s for s in experiment_matrix() if s.name == args.name]
        if not matches:
            raise SystemExit(f"unknown run {args.name!r}")
        spec = matches[0]
    result = train_and_evaluate(spec, force=args.force)
    print(json.dumps({"name": spec.name, "dev": result["dev"], "test_map": result["test"]["map"]}, indent=2))


if __name__ == "__main__":
    main()
