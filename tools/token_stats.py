"""Feature statistics behind the normalization question (no training needed).

For dev images it measures: DINOv2 per-token norms before/after DINOv2's own final LayerNorm (outlier tokens),
per-channel scale spread of the tokens the adapter consumes, the scale of YOLOv8n's P5/SPPF features, and the
context-to-P5 scale ratio each adapter arm produces at initialization.

    python -m tools.token_stats
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import load_checkpoint

from nafuse import PROJECT_ROOT, RESULTS_ROOT
from nafuse.adapter import ARMS, FusionAdapter
from nafuse.data import VOC_DEV_DIR, VOC_DIR
from nafuse.model import FusedDetectionModel


def _load(path, size=640):
    img = cv2.imread(str(path))[:, :, ::-1]
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (round(w * s / 32) * 32, round(h * s / 32) * 32), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255


@torch.no_grad()
def main(n_images: int = 400, seed: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    files = sorted((VOC_DIR / VOC_DEV_DIR).glob("*.jpg"))
    files = [files[i] for i in rng.choice(len(files), n_images, replace=False)]

    weights, _ = load_checkpoint(PROJECT_ROOT / "yolov8n.pt")
    yolo = FusedDetectionModel(weights.yaml, nc=80, verbose=False)
    yolo.load(weights, verbose=False)
    yolo = yolo.to(device).eval()
    p5_holder = {}
    hook = yolo.model[9].register_forward_hook(lambda m, i, o: p5_holder.__setitem__("p5", o))

    torch.manual_seed(seed)
    adapters = {arm: FusionAdapter(arm).to(device).eval() for arm in ARMS}
    dino = adapters["lngn"].dino

    pre_norms, post_norms, post_tokens, p5_rms, ratios = [], [], [], [], {arm: [] for arm in ARMS}
    for f in files:
        x = _load(f).unsqueeze(0).to(device)
        raw = dino.tokens(x, final_norm=False)
        post = dino.tokens(x, final_norm=True)
        pre_norms.append(raw.flatten(2).norm(dim=1).flatten().cpu())
        post_norms.append(post.flatten(2).norm(dim=1).flatten().cpu())
        post_tokens.append(post.flatten(2).squeeze(0).T.cpu())
        yolo(x)
        p5 = p5_holder["p5"]
        p5_rms.append(float(p5.pow(2).mean().sqrt()))
        for arm, adapter in adapters.items():
            ctx = adapter(x, p5.shape[-2:])
            ratios[arm].append(float(ctx.pow(2).mean().sqrt() / p5.pow(2).mean().sqrt()))
    hook.remove()

    pre, post = torch.cat(pre_norms).numpy(), torch.cat(post_norms).numpy()
    tokens = torch.cat(post_tokens)
    ch_std = tokens.std(0).numpy()

    def outliers(norms):
        q1, q3 = np.percentile(norms, [25, 75])
        return float((norms > q3 + 3 * (q3 - q1)).mean())

    def summary(norms):
        return {
            "median": float(np.median(norms)),
            "p99": float(np.percentile(norms, 99)),
            "max": float(norms.max()),
            "max_over_median": float(norms.max() / np.median(norms)),
            "outlier_fraction_iqr3": outliers(norms),
        }

    out = {
        "n_images": n_images,
        "dino_token_norm_pre_final_norm": summary(pre),
        "dino_token_norm_post_final_norm": summary(post),
        "dino_post_norm_rms": float(tokens.pow(2).mean().sqrt()),
        "dino_channel_std": {
            "min": float(ch_std.min()),
            "median": float(np.median(ch_std)),
            "max": float(ch_std.max()),
            "max_over_min": float(ch_std.max() / ch_std.min()),
        },
        "yolo_p5_rms_mean": float(np.mean(p5_rms)),
        "init_ctx_over_p5_rms": {arm: float(np.mean(v)) for arm, v in ratios.items()},
        "hist": {
            "pre_final_norm": np.histogram(pre, bins=80)[0].tolist(),
            "pre_edges": np.histogram(pre, bins=80)[1].tolist(),
            "post_final_norm": np.histogram(post, bins=80)[0].tolist(),
            "post_edges": np.histogram(post, bins=80)[1].tolist(),
            "channel_std": np.sort(ch_std).tolist(),
        },
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "token_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "hist"}, indent=2))


if __name__ == "__main__":
    main()
