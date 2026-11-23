"""Side-by-side test images where fusion helps / hurts most (from saved per-image records; no GPU needed).

    python -m tools.qualitative            # baseline-s0 vs lngn-s0 of the main group
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics.utils import YAML

from nafuse import RESULTS_ROOT, RUNS_ROOT
from nafuse.data import VOC_DIR, VOC_TEST_DIR, dataset_yaml

CONF = 0.25


def load_records(path):
    d = np.load(path)
    po, to = d["conf_offsets"], d["target_cls_offsets"]
    out = {}
    for i, im in enumerate(d["im"]):
        p, t = slice(po[i], po[i + 1]), slice(to[i], to[i + 1])
        keep = d["conf"][p] >= CONF
        out[str(im)] = {
            "boxes": d["pred_boxes"][p][keep],
            "conf": d["conf"][p][keep],
            "cls": d["pred_cls"][p][keep],
            "tp": d["tp"][p][keep][:, 0] if keep.any() else np.zeros(0, bool),
            "gt_boxes": d["target_boxes"][t],
            "gt_cls": d["target_cls"][t],
        }
    return out


def score(r) -> int:
    return int(r["tp"].sum()) - int((~r["tp"].astype(bool)).sum())


def draw(ax, image, r, names, title):
    ax.imshow(image)
    for b in r["gt_boxes"]:
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", ls="--", lw=1.2))
    for b, c, k, tp in zip(r["boxes"], r["conf"], r["cls"], r["tp"]):
        color = "#2BD47D" if tp else "#FF4D4D"
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=1.8))
        ax.text(b[0], b[1] - 2, f"{names[int(k)]} {c:.2f}", color="black", fontsize=6.5, backgroundcolor=color, va="bottom")
    tp = int(r["tp"].sum())
    ax.set_title(f"{title}: TP {tp}  FP {len(r['tp']) - tp}", fontsize=9)
    ax.axis("off")


def make_figure(base, fused, ims, names, label, path):
    fig, axes = plt.subplots(2, len(ims), figsize=(4.2 * len(ims), 6.6), squeeze=False)
    for j, im in enumerate(ims):
        image = Image.open(VOC_DIR / VOC_TEST_DIR / im).convert("RGB")
        draw(axes[0, j], image, base[im], names, "Baseline")
        draw(axes[1, j], image, fused[im], names, label)
    fig.text(0.01, 0.01, "white dashed = ground truth · green = correct (IoU≥0.5) · red = false positive · conf ≥ 0.25", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build(
    base_name: str = "voc-yolov8n-baseline-s0", fused_name: str = "voc-yolov8n-lngn-s0", n_help: int = 3, n_hurt: int = 2
) -> dict | None:
    paths = [RUNS_ROOT / "main" / n / "test" / "records.npz" for n in (base_name, fused_name)]
    if not all(p.exists() for p in paths):
        return None
    names = YAML.load(dataset_yaml("voc"))["names"]
    base, fused = (load_records(p) for p in paths)
    candidates = [im for im in base if 1 <= len(base[im]["gt_cls"]) <= 6 and im in fused]
    diff = {im: score(fused[im]) - score(base[im]) for im in candidates}
    ranked = sorted(candidates, key=lambda im: diff[im])
    helps, hurts = ranked[::-1][:n_help], ranked[:n_hurt]
    fig_dir = RESULTS_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    make_figure(base, fused, helps, names, "LN+GN fusion", fig_dir / "qualitative_helps.png")
    make_figure(base, fused, hurts, names, "LN+GN fusion", fig_dir / "qualitative_hurts.png")
    out = {
        "base": base_name,
        "fused": fused_name,
        "helps": helps,
        "hurts": hurts,
        "images_better": int(sum(d > 0 for d in diff.values())),
        "images_worse": int(sum(d < 0 for d in diff.values())),
        "images_compared": len(diff),
        "figures": [str(fig_dir / "qualitative_helps.png"), str(fig_dir / "qualitative_hurts.png")],
    }
    (RESULTS_ROOT / "qualitative.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="voc-yolov8n-baseline-s0")
    parser.add_argument("--fused", default="voc-yolov8n-lngn-s0")
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.fused), indent=2))


if __name__ == "__main__":
    main()
