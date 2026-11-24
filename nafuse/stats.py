"""Result aggregation and image-level paired bootstrap for mAP differences.

The bootstrap re-implements Ultralytics' ``ap_per_class``/``compute_ap`` (101-point COCO interpolation) on the
per-image match records saved at test time. Resampling images with replacement only re-weights predictions, so
the confidence ordering is computed once and each replicate is a weighted cumulative sum.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import RUNS_ROOT

_X = np.linspace(0, 1, 101)
_EPS = 1e-16


def _ap_from_curves(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapezoid(np.interp(_X, mrec, mpre), _X))


class PooledDetections:
    """All test-set detections of one model, pre-sorted per class for fast re-weighted AP."""

    def __init__(self, records_npz: str | Path):
        d = np.load(records_npz)
        self.images = d["im"]
        n_img = len(self.images)
        img_of_pred = np.repeat(np.arange(n_img), np.diff(d["conf_offsets"]))
        img_of_gt = np.repeat(np.arange(n_img), np.diff(d["target_cls_offsets"]))
        conf, pred_cls, target_cls = d["conf"], d["pred_cls"].astype(int), d["target_cls"].astype(int)
        tp = d["tp"].reshape(len(conf), -1)
        nc = int(max(pred_cls.max(initial=0), target_cls.max(initial=0)) + 1)
        self.gt_counts = np.zeros((n_img, nc), dtype=np.float32)
        np.add.at(self.gt_counts, (img_of_gt, target_cls), 1.0)
        self.classes = np.unique(target_cls)
        order = np.argsort(-conf)
        self.per_class = {}
        for c in self.classes:
            sel = order[pred_cls[order] == c]
            self.per_class[int(c)] = (img_of_pred[sel].astype(np.int32), tp[sel].astype(np.float32))

    def ap(self, weights: np.ndarray | None = None) -> np.ndarray:
        """(n_classes, 10) AP matrix; ``weights`` are per-image bootstrap multiplicities."""
        out = []
        for c in self.classes:
            img, tp = self.per_class[int(c)]
            if weights is None:
                n_l = self.gt_counts[:, c].sum()
                w = None
            else:
                n_l = float(weights @ self.gt_counts[:, c])
                w = weights[img]
                keep = w > 0
                img, tp, w = img[keep], tp[keep], w[keep]
            if n_l == 0:
                continue
            if len(tp) == 0:
                out.append(np.zeros(tp.shape[1] if tp.ndim == 2 else 10))
                continue
            tpw = tp if w is None else tp * w[:, None]
            fpw = (1.0 - tp) if w is None else (1.0 - tp) * w[:, None]
            tpc, fpc = tpw.cumsum(0), fpw.cumsum(0)
            recall = tpc / (n_l + _EPS)
            precision = tpc / (tpc + fpc)
            out.append(np.array([_ap_from_curves(recall[:, j], precision[:, j]) for j in range(tp.shape[1])]))
        return np.array(out)

    def map(self, weights: np.ndarray | None = None) -> tuple[float, float]:
        ap = self.ap(weights)
        return float(ap[:, 0].mean()), float(ap.mean())


def _replicate_weights(n_img: int, seed: int, b: int) -> np.ndarray:
    rng = np.random.default_rng([seed, b])
    return np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(np.float32)


def _model_bootstrap(args) -> tuple[str, np.ndarray, tuple[float, float], np.ndarray]:
    path, n_boot, seed = args
    pooled = PooledDetections(path)
    n_img = len(pooled.images)
    boots = np.array([pooled.map(_replicate_weights(n_img, seed, b)) for b in range(n_boot)])
    return str(path), pooled.images, pooled.map(), boots


def bootstrap_maps(paths: list[str | Path], n_boot: int = 500, seed: int = 0, workers: int = 4) -> dict[str, dict]:
    """(mAP50, mAP50-95) for every model on the same image replicates, so any paired delta is a subtraction."""
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_model_bootstrap, [(p, n_boot, seed) for p in paths]))
    reference = results[0][1]
    for path, images, _, _ in results:
        if not np.array_equal(images, reference):
            raise ValueError(f"{path}: test images differ in content/order; paired resampling impossible")
    return {path: {"point": np.array(point), "boot": boots} for path, _, point, boots in results}


def paired_delta(boot: dict[str, dict], pairs: list[tuple[str, str]]) -> dict:
    """Mean (B - A) over seed-matched pairs with a 95% percentile CI from shared image replicates."""
    point = np.mean([boot[b]["point"] - boot[a]["point"] for a, b in pairs], axis=0)
    reps = np.mean([boot[b]["boot"] - boot[a]["boot"] for a, b in pairs], axis=0)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    lo90, hi90 = np.percentile(reps[:, 1], [5, 95])
    return {
        "n_pairs": len(pairs),
        "n_boot": len(reps),
        "delta_map50": float(point[0]),
        "delta_map": float(point[1]),
        "ci_map50": [float(lo[0]), float(hi[0])],
        "ci_map": [float(lo[1]), float(hi[1])],
        "ci90_map": [float(lo90), float(hi90)],  # for equivalence (TOST) at alpha = 0.05
        "p_le_zero_map": float((reps[:, 1] <= 0).mean()),
        "p_two_sided_map": float(min(1.0, 2 * min((reps[:, 1] <= 0).mean(), (reps[:, 1] >= 0).mean()))),
    }


def load_results(root: Path = RUNS_ROOT) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(root).glob("*/*/result.json")):
        r = json.loads(path.read_text())
        spec, test, dev = r["spec"], r["test"], r["dev"]
        rows.append(
            {
                **{k: spec[k] for k in ("name", "arm", "seed", "group", "dataset", "model", "dino_res", "epochs")},
                "run_dir": str(path.parent),
                "records": str(path.parent / "test" / "records.npz"),
                "best_epoch": r["best_epoch"],
                "train_hours": r["train_hours"],
                **{f"dev_{k}": v for k, v in dev.items()},
                **{f"test_{k}": test[k] for k in ("precision", "recall", "map50", "map", "map_small", "map_medium", "map_large")},
                **r["params"],
                "dev_curve": r.get("dev_curve"),
                "per_class_map": test.get("per_class_map"),
            }
        )
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n"))
    return out.reset_index()
