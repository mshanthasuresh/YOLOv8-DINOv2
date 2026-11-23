"""Held-out evaluation of a finished checkpoint (overall, COCO size bins, per class, per-image records)."""

from __future__ import annotations

from pathlib import Path

from .trainer import RecordingValidator


def evaluate_checkpoint(
    weights: Path, data_yaml: Path, split: str, out_dir: Path, batch: int = 32, device=0, records: bool = True
) -> dict:
    out_dir = Path(out_dir)
    validator = RecordingValidator(
        args={
            "model": str(weights),
            "data": str(data_yaml),
            "split": split,
            "imgsz": 640,
            "batch": batch,
            "conf": 0.001,
            "iou": 0.7,
            "save_json": True,
            "plots": False,
            "verbose": False,
            "device": device,
            "workers": 4,
            "channels_last": False,
            "project": str(out_dir.parent),
            "name": out_dir.name,
            "exist_ok": True,
        }
    )
    stats = validator()
    if records:
        validator.save_records(out_dir / "records.npz")
    (out_dir / "predictions.json").unlink(missing_ok=True)  # COCO-format dump is large and fully covered by records
    box = validator.metrics.box
    names = validator.names
    return {
        "precision": stats["metrics/precision(B)"],
        "recall": stats["metrics/recall(B)"],
        "map50": stats["metrics/mAP50(B)"],
        "map": stats["metrics/mAP50-95(B)"],
        "map_small": stats.get("metrics/mAP_small(B)"),
        "map_medium": stats.get("metrics/mAP_medium(B)"),
        "map_large": stats.get("metrics/mAP_large(B)"),
        "per_class_map": {names[int(c)]: float(v) for c, v in zip(box.ap_class_index, box.ap)},
        "per_class_map50": {names[int(c)]: float(v) for c, v in zip(box.ap_class_index, box.ap50)},
        "images": int(validator.seen),
        "speed_ms": dict(validator.speed),
    }
