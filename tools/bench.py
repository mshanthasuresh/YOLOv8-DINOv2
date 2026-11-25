"""Measure latency / parameters / FLOPs for every architecture (run with the GPU otherwise idle).

    python -m tools.bench
"""

from __future__ import annotations

import argparse

from nafuse import RESULTS_ROOT
from nafuse.bench import run_benchmarks, run_fast_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="batch-1 latency: eager vs CUDA-graph replay")
    args = parser.parse_args()
    if args.fast:
        for row in run_fast_benchmarks(RESULTS_ROOT / "bench_fast.json"):
            print(
                f"{row['model']:8s} {row['arm']:9s} res={row['dino_res']!s:5s} eager={row['eager_ms']:.2f}ms "
                f"graph={row['graph_ms']:.2f}ms  max|diff|={row['max_abs_diff']:.1e}"
            )
        return
    for row in run_benchmarks(RESULTS_ROOT / "bench.json", n=args.n, rounds=args.rounds):
        print(
            f"{row['model']:8s} {row['arm']:9s} res={row['dino_res']!s:5s} "
            f"latency={row['latency_ms']:.2f}ms (dino {row['dino_latency_ms']:.2f}) "
            f"b32={row['ms_per_image_batched']:.2f}ms/img GFLOPs={row['gflops_total']:.1f} "
            f"(dino {row['gflops_dino']:.1f}) params={row['params_total'] / 1e6:.2f}M"
        )


if __name__ == "__main__":
    main()
