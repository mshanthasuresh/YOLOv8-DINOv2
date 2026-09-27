from __future__ import annotations

import argparse
import contextlib
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def load_dino(device: torch.device):
    with warnings.catch_warnings(), open(os.devnull, "w", encoding="utf-8") as sink:
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14"
            )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DINOv2 semantic extraction before YOLO detection."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs" / "dino_inference" / "prediction.jpg"),
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if args.device != "cpu" else "cpu")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    dino = load_dino(device)
    transform = T.Compose([
        T.Resize((518, 518), antialias=True),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = dino(tensor)
    embedding_norm = float(torch.linalg.vector_norm(embedding).item())

    detector = YOLO(args.weights)
    result = detector.predict(
        source=frame,
        device=0 if args.device != "cpu" else "cpu",
        imgsz=args.imgsz,
        verbose=False,
    )[0]
    annotated = result.plot()
    cv2.putText(
        annotated,
        f"DINOv2 embedding norm: {embedding_norm:.3f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(output_path), annotated)
    print(f"DINO device: {device}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Embedding norm: {embedding_norm:.3f}")
    print(f"Detections: {len(result.boxes)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
