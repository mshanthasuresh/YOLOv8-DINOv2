"""End-to-end cost of every architecture: latency (incl. DINOv2), parameters and FLOPs.

Laptop GPUs drift with temperature, so configurations are timed round-robin over several rounds and the
median across rounds is reported; FLOPs are the hardware-independent cost measure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .dino import FrozenDINO, load_backbone
from .fast import GraphedModel, first_tensor
from .model import build_model

CONFIGS = [("yolov8n.yaml", None), ("yolov8s.yaml", None)]
CONFIGS += [("yolov8n.yaml", {"arm": arm, "resolution": 518}) for arm in ("nonorm", "ln", "gn", "lngn", "gated")]
CONFIGS += [("yolov8n.yaml", {"arm": "lngn", "resolution": r}) for r in (224, 336, 448)]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def _median_ms(fn, n: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def _gflops(fn) -> float:
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        fn()
    return counter.get_total_flops() / 1e9


def run_benchmarks(out_path: Path, n: int = 50, rounds: int = 5, imgsz: int = 640, batch: int = 32) -> list[dict]:
    device = "cuda"
    x1 = torch.rand(1, 3, imgsz, imgsz, device=device)
    xb = torch.rand(batch, 3, imgsz, imgsz, device=device, dtype=torch.half)
    entries = []
    for cfg, fusion in CONFIGS:
        model = build_model(cfg, nc=20, fusion=fusion).to(device).eval()
        model.fuse(verbose=False)
        row = {
            "model": cfg.replace(".yaml", ""),
            "arm": fusion["arm"] if fusion else "baseline",
            "dino_res": fusion["resolution"] if fusion else None,
            "params_detector": sum(p.numel() for p in model.parameters()),
            "gflops_total": _gflops(lambda m=model: m(x1)),
            "params_dino": 0,
            "gflops_dino": 0.0,
        }
        dino = None
        if fusion:
            dino = FrozenDINO(resolution=fusion["resolution"])
            row["params_dino"] = sum(p.numel() for p in load_backbone(dino.name, device).parameters())
            row["gflops_dino"] = _gflops(lambda d=dino: d.tokens(x1))
        row["params_total"] = row["params_detector"] + row["params_dino"]
        entries.append((row, model.half(), dino))

    samples = {i: {"latency": [], "dino": [], "batched": []} for i in range(len(entries))}
    for _ in range(rounds):
        for i, (row, model, dino) in enumerate(entries):
            samples[i]["latency"].append(_median_ms(lambda m=model: m(x1.half()), n, 10))
            samples[i]["batched"].append(_median_ms(lambda m=model: m(xb), max(n // 10, 3), 2) / batch)
            if dino is not None:
                samples[i]["dino"].append(_median_ms(lambda d=dino: d.tokens(x1.half()), n, 10))

    rows = []
    for i, (row, _, _) in enumerate(entries):
        row["latency_ms"] = float(np.median(samples[i]["latency"]))
        row["latency_ms_rounds"] = samples[i]["latency"]
        row["dino_latency_ms"] = float(np.median(samples[i]["dino"])) if samples[i]["dino"] else 0.0
        row["ms_per_image_batched"] = float(np.median(samples[i]["batched"]))
        rows.append(row)

    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol": f"{imgsz}x{imgsz}, FP16, Conv+BN fused, forward pass incl. DINOv2 preprocessing (no NMS), "
        f"CUDA-synchronized. latency_ms: median over {rounds} round-robin rounds of the median of {n} batch-1 "
        f"iterations. ms_per_image_batched: batch-{batch} time / {batch}. FLOPs: torch FlopCounterMode.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    return rows


FAST_CONFIGS = [("yolov8n.yaml", None), ("yolov8s.yaml", None)]
FAST_CONFIGS += [("yolov8n.yaml", {"arm": "lngn", "resolution": r}) for r in (224, 336, 448, 518)]


def run_fast_benchmarks(out_path: Path, n: int = 200, rounds: int = 5, imgsz: int = 640) -> list[dict]:
    """Batch-1 latency, eager vs CUDA-graph replay, timed round-robin over all models like run_benchmarks."""
    x = torch.rand(1, 3, imgsz, imgsz, device="cuda", dtype=torch.half)
    entries = []
    for cfg, fusion in FAST_CONFIGS:
        model = build_model(cfg, nc=20, fusion=fusion).to("cuda").eval()
        model.fuse(verbose=False)
        model.half()
        with torch.inference_mode():
            reference = first_tensor(model(x)).clone()
        graphed = GraphedModel(model, x)
        row = {
            "model": cfg.replace(".yaml", ""),
            "arm": fusion["arm"] if fusion else "baseline",
            "dino_res": fusion["resolution"] if fusion else None,
            "max_abs_diff": float((first_tensor(graphed(x)) - reference).abs().max()),
        }
        entries.append((row, model, graphed))
    samples = {i: {"eager": [], "graph": []} for i in range(len(entries))}
    for _ in range(rounds):
        for i, (_, model, graphed) in enumerate(entries):
            samples[i]["eager"].append(_median_ms(lambda m=model: m(x), n // 4, 10))
            samples[i]["graph"].append(_median_ms(lambda g=graphed: g(x), n, 20))
    rows = []
    for i, (row, _, _) in enumerate(entries):
        row["eager_ms"] = float(np.median(samples[i]["eager"]))
        row["graph_ms"] = float(np.median(samples[i]["graph"]))
        rows.append(row)
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol": f"batch 1, {imgsz}x{imgsz}, FP16 detector, Conv+BN fused, forward incl. DINOv2 preprocessing (no NMS). "
        f"Median over {rounds} round-robin rounds of the median of {n} graph / {n // 4} eager iterations. "
        "max_abs_diff: largest |difference| between the graph replay and the eager head output.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    return rows
