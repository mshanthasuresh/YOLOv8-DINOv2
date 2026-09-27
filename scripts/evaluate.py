from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained detector."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "coco8.yaml"))
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="evaluation")
    args = parser.parse_args()

    if args.fused:
        from dino_modules import register_modules

        register_modules()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
    )
    summary = {
        "weights": args.weights,
        "data": args.data,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    output = ROOT / "runs" / args.name / "metrics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
