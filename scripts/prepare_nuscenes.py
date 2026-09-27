from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def project_box(box, intrinsic: np.ndarray, width: int, height: int):
    corners = box.corners()
    depths = corners[2]
    if np.all(depths <= 0.1):
        return None
    points = intrinsic @ corners
    points = points[:2] / np.maximum(points[2:3], 1e-6)
    x1, y1 = np.maximum(points.min(axis=1), [0, 0])
    x2, y2 = np.minimum(points.max(axis=1), [width - 1, height - 1])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert nuScenes CAM_FRONT boxes to YOLO labels.")
    parser.add_argument("--root", required=True, help="Directory containing nuScenes tables and samples.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", default="data/nuscenes_yolo")
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a smoke test; 0 means all samples.")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    output = Path(args.output)
    image_dir = output / "images" / "val"
    label_dir = output / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted({sample_annotation["category_name"].split(".")[0] for sample_annotation in nusc.sample_annotation})
    class_ids = {name: index for index, name in enumerate(class_names)}

    count = 0
    for sample in nusc.sample:
        if args.limit and count >= args.limit:
            break
        token = sample["data"].get("CAM_FRONT")
        if not token:
            continue
        sample_data = nusc.get("sample_data", token)
        camera = nusc.get_sample_data(token)
        image_path, boxes, intrinsic = camera
        image = __import__("cv2").imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        destination_name = f"{sample['token']}.jpg"
        shutil.copy2(image_path, image_dir / destination_name)
        labels = []
        for box in boxes:
            category = box.name.split(".")[0]
            if category not in class_ids:
                continue
            projected = project_box(box, np.asarray(intrinsic), width, height)
            if projected is None:
                continue
            x1, y1, x2, y2 = projected
            labels.append(f"{class_ids[category]} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        (label_dir / destination_name.replace(".jpg", ".txt")).write_text("\n".join(labels) + "\n", encoding="utf-8")
        count += 1

    names = "\n".join(f"  {index}: {name}" for name, index in class_ids.items())
    (output / "data.yaml").write_text(f"path: {output.resolve()}\ntrain: images/val\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    print(f"Prepared {count} nuScenes camera images in {output}")
    print(f"Classes: {len(class_ids)}")


if __name__ == "__main__":
    main()