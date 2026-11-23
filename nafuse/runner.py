"""Train one run, select its checkpoint on the dev split, then evaluate it once on the held-out test split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from . import RUNS_ROOT
from .adapter import SplitConv2d
from .data import dataset_yaml
from .evaluate import evaluate_checkpoint
from .specs import RunSpec
from .trainer import FusionTrainer

TRAIN_DEFAULTS = {
    "imgsz": 640,
    "batch": 16,
    "workers": 8,
    "device": 0,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 1.0,
    "close_mosaic": 2,
    "amp": True,
    "channels_last": False,  # measured 5x slower training steps on the RTX A3000 laptop GPU
    "deterministic": False,
    "patience": 0,
    "plots": False,
    "val": True,
    "exist_ok": True,
    "verbose": True,
}


def run_dir(spec: RunSpec) -> Path:
    return RUNS_ROOT / spec.group / spec.name


def dev_history(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    df["fitness"] = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return df


def param_counts(model) -> dict:
    model = unwrap_model(model)
    fusion = getattr(model, "fusion", None)
    added_neck = sum(
        (c.dino_slice[1] - c.dino_slice[0]) * c.out_channels * c.kernel_size[0] * c.kernel_size[1]
        for c in model.modules()
        if isinstance(c, SplitConv2d)
    )
    return {
        "detector_params": sum(p.numel() for p in model.parameters()),
        "adapter_params": sum(p.numel() for p in fusion.parameters()) if fusion is not None else 0,
        "neck_dino_params": added_neck,
    }


def train_and_evaluate(spec: RunSpec, force: bool = False) -> dict:
    out = run_dir(spec)
    result_path = out / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / "weights" / "last.pt"
    results_path = out / "results.csv"
    resuming = not force and last_path.is_file() and results_path.is_file()
    previous_epoch = -1
    previous_seconds = 0.0
    if resuming:
        _, checkpoint = load_checkpoint(last_path, device="cpu")
        checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
        del checkpoint
        history_before = pd.read_csv(results_path)
        history_before = history_before[history_before["epoch"] <= checkpoint_epoch]
        history_before.to_csv(results_path, index=False)
        previous_epoch = checkpoint_epoch
        previous_seconds = float(history_before.iloc[-1]["time"]) if not history_before.empty else 0.0
        if previous_epoch >= spec.epochs:
            resuming = False
    if not resuming and previous_epoch < spec.epochs:
        (out / "diagnostics.jsonl").unlink(missing_ok=True)

    data_yaml = dataset_yaml(spec.dataset)
    overrides = {
        **TRAIN_DEFAULTS,
        **spec.train,
        "model": str(last_path) if resuming else f"{spec.model}.pt",
        "data": str(data_yaml),
        "epochs": spec.epochs,
        "seed": spec.seed,
        "project": str(out.parent),
        "name": out.name,
        "resume": str(last_path) if resuming else False,
    }
    start = time.time()
    if previous_epoch < spec.epochs:
        trainer = FusionTrainer(overrides=overrides, fusion=spec.fusion_cfg())
        trainer.train()
        counts = param_counts(trainer.model)
        layout = getattr(unwrap_model(trainer.model), "fusion_layout", [])
        del trainer
        torch.cuda.empty_cache()
        if resuming:
            history = pd.read_csv(results_path)
            history.loc[history["epoch"] > previous_epoch, "time"] += previous_seconds
            history.to_csv(results_path, index=False)
    else:
        LOGGER.info("Training checkpoint is complete; continuing directly to held-out evaluation")
        model, _ = load_checkpoint(last_path, device="cpu")
        counts = param_counts(model)
        layout = getattr(model, "fusion_layout", [])
        del model
    train_hours = (previous_seconds + time.time() - start) / 3600

    history = dev_history(results_path)
    best = history.loc[history["fitness"].idxmax()]
    best_path = out / "weights" / "best.pt"
    test = evaluate_checkpoint(best_path if best_path.is_file() else last_path, data_yaml, "test", out / "test")
    result = {
        "spec": spec.to_dict(),
        "train_hours": train_hours,
        "best_epoch": int(best["epoch"]),
        "dev": {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]),
            "map": float(best["metrics/mAP50-95(B)"]),
        },
        "dev_curve": history["metrics/mAP50-95(B)"].round(5).tolist(),
        "test": test,
        "params": counts,
        "fusion_layout": layout,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(result_path)
    return result
