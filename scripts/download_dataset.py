import argparse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"

parser = argparse.ArgumentParser(description="Prepare a small YOLO dataset.")
parser.add_argument("--dataset", choices=["coco8"], default="coco8")
args = parser.parse_args()

if args.dataset == "coco8":
    destination = ROOT / "data" / "coco8"
    archive = ROOT / "data" / "coco8.zip"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public COCO8 from {COCO8_URL}")
    urllib.request.urlretrieve(COCO8_URL, archive)
    with zipfile.ZipFile(archive) as compressed:
        compressed.extractall(ROOT / "data")
    archive.unlink()
    names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    yaml = "path: coco8\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    (ROOT / "data" / "coco8.yaml").write_text(yaml + "\n", encoding="utf-8")
    print("COCO8 is ready at data/coco8. Use data/coco8.yaml for training.")
