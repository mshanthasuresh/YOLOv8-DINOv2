from pathlib import Path
import argparse

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(ROOT / "data" / "coco8.yaml"))
parser.add_argument("--model", default="yolov8n.pt")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--fused", action="store_true")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

if args.fused:
    from dino_modules import register_modules

    register_modules()
    fused_config = ROOT / "configs" / "yolov8-dino.yaml"
    model = YOLO(str(fused_config))
    pretrained = YOLO(args.model).model.state_dict()
    fused_state = model.model.state_dict()
    transferred = 0
    for key, value in fused_state.items():
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        layer_index = int(parts[1])
        if 1 <= layer_index <= 10:
            source_index = layer_index - 1
        elif 13 <= layer_index <= 25:
            source_index = layer_index - 3
        else:
            continue
        source_key = f"model.{source_index}.{parts[2]}"
        source_value = pretrained.get(source_key)
        if source_value is not None and source_value.shape == value.shape:
            fused_state[key] = source_value
            transferred += 1
        elif (
            source_value is not None
            and value.ndim == 4
            and source_value.ndim == 4
            and value.shape[0] == source_value.shape[0]
            and value.shape[1] == source_value.shape[1] + 3
            and value.shape[2:] == source_value.shape[2:]
        ):
            padded = torch.zeros_like(value)
            context_index = (
                0 if layer_index == 15 else source_value.shape[1] // 3
            )
            padded[:, :context_index] = source_value[:, :context_index]
            padded[:, context_index + 3:] = source_value[:, context_index:]
            torch.nn.init.normal_(
                padded[:, context_index:context_index + 3], mean=0.0, std=0.01
            )
            fused_state[key] = padded
            transferred += 1
    model.model.load_state_dict(fused_state)
    print(f"Transferred {transferred} compatible tensors from {args.model}")

    def freeze_dinov2(trainer):
        dinov2 = trainer.model.model[11].model
        dinov2.requires_grad_(False)
        dinov2.eval()

    def log_adapter_norm(trainer):
        projector = trainer.model.model[11].projector
        weight_norm = float(projector.weight.detach().norm())
        bias_norm = float(projector.bias.detach().norm())
        print(
            f"DINO adapter epoch {trainer.epoch + 1}: "
            f"weight_norm={weight_norm:.6g}, bias_norm={bias_norm:.6g}"
        )

    model.add_callback("on_train_start", freeze_dinov2)
    model.add_callback("on_train_epoch_end", log_adapter_norm)
else:
    model = YOLO(args.model)

model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    amp=False,
    pretrained=not args.fused,
    project=str(ROOT / "runs"),
    name=(
        "fused_yolov8"
        if args.fused
        else "baseline_yolov8" if args.baseline else "yolov8_training"
    ),
)
