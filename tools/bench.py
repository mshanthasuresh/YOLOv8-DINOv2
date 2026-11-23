"""Measure latency / parameters / FLOPs for every architecture (run with the GPU otherwise idle).

    python -m tools.bench
"""

from __future__ import annotations

import argparse

from nafuse import RESULTS_ROOT
from nafuse.bench import run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    for row in run_benchmarks(RESULTS_ROOT / "bench.json", n=args.n, rounds=args.rounds):
        print(
            f"{row['model']:8s} {row['arm']:9s} res={row['dino_res']!s:5s} "
            f"latency={row['latency_ms']:.2f}ms (dino {row['dino_latency_ms']:.2f}) "
            f"b32={row['ms_per_image_batched']:.2f}ms/img GFLOPs={row['gflops_total']:.1f} "
            f"(dino {row['gflops_dino']:.1f}) params={row['params_total'] / 1e6:.2f}M"
        )


if __name__ == "__main__":
    main()
