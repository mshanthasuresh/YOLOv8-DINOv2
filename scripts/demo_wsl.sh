#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-wsl/bin/python}"
INPUT="${1:-$ROOT/data/coco8/images/val/000000000036.jpg}"
OUTPUT="${2:-$ROOT/runs/dino_inference/wsl_demo.jpg}"

if [[ ! -x "$PYTHON" ]]; then
    echo "WSL environment not found. Run the setup commands in README.md first." >&2
    exit 1
fi

if [[ -f "$ROOT/runs/baseline_yolov8/weights/best.pt" ]]; then
    WEIGHTS="$ROOT/runs/baseline_yolov8/weights/best.pt"
elif [[ -f "$ROOT/../yolov8n.pt" ]]; then
    WEIGHTS="$ROOT/../yolov8n.pt"
else
    WEIGHTS="yolov8n.pt"
fi

echo "Using Python: $PYTHON"
"$PYTHON" -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
"$PYTHON" scripts/dino_inference.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --weights "$WEIGHTS" \
    --device 0 \
    --imgsz 640