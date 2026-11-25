"""Aggregate all finished runs into tables, figures and results/summary.json (consumed by make_slides).

    python -m tools.analyze           # full: includes paired image-bootstrap CIs (slow, CPU)
    python -m tools.analyze --quick   # everything except the bootstrap
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from nafuse import RESULTS_ROOT
from nafuse.errors import ERRORS, OPERATING_CONF, decompose
from nafuse.specs import ALL_ARMS, CONTROL_ARMS, DEFAULT_RES, FUSION_ARMS
from nafuse.stats import bootstrap_maps, load_results, mean_std, paired_delta
from tools import qualitative

FIG = RESULTS_ROOT / "figures"
COLORS = {
    "baseline": "#6B7B78",
    "nonorm": "#E8A33D",
    "ln": "#9B6FB0",
    "gn": "#4C8DC9",
    "lngn": "#007B73",
    "gated": "#E85E48",
    "yolov8s": "#152E2A",
    "global": "#7CC6BE",
    "gatedlow": "#F29E8E",
}
SHORT = {
    "baseline": "Baseline",
    "nonorm": "No norm",
    "ln": "LN only",
    "gn": "GN only",
    "lngn": "LN+GN",
    "gated": "Gated",
    "global": "Global DINO",
    "gatedlow": "Gated, low init",
}
EQUIV_MARGIN = 0.005  # ±0.5 mAP50-95 points, about the seed-to-seed sd of the baseline
METRICS = [
    "test_map",
    "test_map50",
    "test_precision",
    "test_recall",
    "test_map_small",
    "test_map_medium",
    "test_map_large",
    "dev_map",
    "train_hours",
    "best_epoch",
]
ITERS_PER_EPOCH = math.ceil(14041 / 16)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
    }
)


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _records(df: pd.DataFrame) -> list[dict]:
    keep = [c for c in df.columns if c not in {"dev_curve", "per_class_map"}]
    return json.loads(df[keep].to_json(orient="records"))


def _arm_table(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    agg = mean_std(df, ["arm"], METRICS)
    out = {}
    for _, row in agg.iterrows():
        sub = df[df["arm"] == row["arm"]].sort_values("seed")
        out[row["arm"]] = {
            "n": int(row["n"]),
            **{f"{m}_mean": None if pd.isna(row[f"{m}_mean"]) else float(row[f"{m}_mean"]) for m in METRICS},
            **{f"{m}_std": None if pd.isna(row[f"{m}_std"]) else float(row[f"{m}_std"]) for m in METRICS},
            "seeds": sub["seed"].tolist(),
            "test_map_per_seed": sub["test_map"].round(5).tolist(),
        }
    return out


def fig_main(main: pd.DataFrame, table: dict) -> str | None:
    arms = [a for a in ALL_ARMS if a in table]
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    base = table.get("baseline", {}).get("test_map_mean")
    for i, arm in enumerate(arms):
        t = table[arm]
        mean, std = 100 * t["test_map_mean"], 100 * (t["test_map_std"] or 0)
        ax.bar(i, mean, color=COLORS[arm], alpha=0.85, width=0.65)
        ax.errorbar(i, mean, yerr=std, color="black", capsize=4, lw=1)
        seeds = main[main["arm"] == arm]["test_map"] * 100
        ax.scatter(np.full(len(seeds), i) + np.linspace(-0.12, 0.12, len(seeds)), seeds, s=14, color="black", zorder=3)
        label = f"{mean:.1f}" if base is None or arm == "baseline" else f"{mean:.1f}\n({100 * (t['test_map_mean'] - base):+.1f})"
        ax.text(i, mean + std + 0.4, label, ha="center", va="bottom", fontsize=8.5)
    if base is not None:
        ax.axhline(100 * base, color=COLORS["baseline"], ls="--", lw=1)
    lo = min(100 * table[a]["test_map_mean"] for a in arms)
    ax.set_ylim(max(0, lo - 6), max(100 * table[a]["test_map_mean"] for a in arms) + 4)
    ax.set_xticks(range(len(arms)), [SHORT[a] for a in arms])
    ax.set_ylabel("VOC07 test mAP50-95 (%)")
    n = max(table[a]["n"] for a in arms)
    ax.set_title(f"Held-out test accuracy (mean ± sd over up to {n} seeds; dots = seeds)")
    return _save(fig, "main_map")


def fig_delta_ci(deltas: dict) -> str | None:
    arms = [a for a in ALL_ARMS if a in deltas]
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(arms) + 1.2))
    for i, arm in enumerate(arms):
        d = deltas[arm]
        lo, hi = (100 * v for v in d["ci_map"])
        ax.plot([lo, hi], [i, i], color=COLORS[arm], lw=3, solid_capstyle="round")
        ax.scatter(100 * d["delta_map"], i, color="black", zorder=3, s=22)
        ax.text(hi + 0.08, i, f"{100 * d['delta_map']:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=8.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(arms)), [SHORT[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("Δ mAP50-95 vs baseline (points, seed-paired, 95% image-bootstrap CI)")
    ax.set_title("Is the gain larger than the noise?")
    return _save(fig, "delta_ci")


def _load_diag(run_dir: str) -> pd.DataFrame | None:
    path = Path(run_dir) / "diagnostics.jsonl"
    if not path.exists():
        return None
    df = pd.read_json(path, lines=True)
    df["epochs"] = df["iter"] / ITERS_PER_EPOCH
    return df


def diagnostics_summary(main: pd.DataFrame) -> tuple[dict, str | None]:
    curves = {}
    for arm in [a for a in ALL_ARMS if a != "baseline"]:
        frames = [d for d in (_load_diag(r) for r in main[main["arm"] == arm]["run_dir"]) if d is not None and len(d)]
        if frames:
            curves[arm] = pd.concat(frames).groupby("iter").mean(numeric_only=True).reset_index()
    if not curves:
        return {}, None
    panels = [
        ("L12/dino_share", "DINO share of first neck-conv input", "linear"),
        ("feat_ratio", "RMS(DINO context) / RMS(YOLO P5)", "log"),
        ("grad_ratio", "RMS grad(DINO ctx) / RMS grad(P5)", "log"),
        ("L12/w_rms_dino", "RMS of DINO input weights (L12)", "linear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, (key, title, scale) in zip(axes.flat, panels):
        for arm, c in curves.items():
            if key in c:
                y = c[key].rolling(5, min_periods=1).mean()
                ax.plot(c["epochs"], y, color=COLORS[arm], label=SHORT[arm], lw=1.6)
        ax.set_title(title)
        ax.set_yscale(scale)
        ax.set_xlabel("epoch")
    axes[0, 0].legend(frameon=False, fontsize=8)
    path = _save(fig, "diagnostics")
    summary = {}
    for arm, c in curves.items():
        first, last = c.iloc[: max(1, len(c) // 20)], c.iloc[-max(1, len(c) // 10) :]
        summary[arm] = {
            k: {"start": float(first[k].mean()), "end": float(last[k].mean())}
            for k in ("L12/dino_share", "L21/dino_share", "feat_ratio", "grad_ratio", "L12/w_rms_dino")
            if k in c
        }
    return summary, path


def fig_dev_curves(main: pd.DataFrame) -> str | None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    drawn = False
    for arm in ALL_ARMS:
        curves = [c for c in main[main["arm"] == arm]["dev_curve"] if isinstance(c, list) and c]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        arr = 100 * np.array([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        ax.plot(x, arr.mean(0), color=COLORS[arm], label=SHORT[arm], lw=1.8)
        if len(curves) > 1:
            ax.fill_between(x, arr.min(0), arr.max(0), color=COLORS[arm], alpha=0.12)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev (VOC07 val) mAP50-95 (%)")
    ax.set_title("Learning curves on the selection split")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, "dev_curves")


def fig_size_bins(table: dict) -> str | None:
    arms = [a for a in ALL_ARMS if a in table and table[a].get("test_map_small_mean") is not None]
    if not arms:
        return None
    bins = [("test_map_small", "small\n<32²"), ("test_map_medium", "medium\n32²–96²"), ("test_map_large", "large\n>96²")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        means = [100 * table[arm][f"{k}_mean"] for k, _ in bins]
        stds = [100 * (table[arm][f"{k}_std"] or 0) for k, _ in bins]
        x = np.arange(len(bins)) + (j - (len(arms) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, color=COLORS[arm], label=SHORT[arm], capsize=2, alpha=0.9)
    ax.set_xticks(range(len(bins)), [b for _, b in bins])
    ax.set_ylabel("COCO-style AP50-95 (%)")
    ax.set_title("Where does the context help? (object size, VOC07 test)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    return _save(fig, "size_bins")


def fig_per_class(main: pd.DataFrame, arm: str = "lngn") -> tuple[dict, str | None]:
    def mean_pc(a):
        rows = [r for r in main[main["arm"] == a]["per_class_map"] if isinstance(r, dict)]
        return pd.DataFrame(rows).mean() if rows else None

    base, fused = mean_pc("baseline"), mean_pc(arm)
    if base is None or fused is None:
        return {}, None
    delta = (100 * (fused - base)).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(delta.index, delta.values, color=[COLORS[arm] if v >= 0 else COLORS["gated"] for v in delta.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Δ AP50-95 ({SHORT[arm]} − baseline), mean over seeds (%)")
    ax.set_title("Per-class change")
    return delta.round(3).to_dict(), _save(fig, "per_class_delta")


def fig_tokens() -> str | None:
    path = RESULTS_ROOT / "token_stats.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text())
    h = t["hist"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for key, edges, color, label in (
        ("pre_final_norm", "pre_edges", COLORS["nonorm"], "before DINOv2 final LN"),
        ("post_final_norm", "post_edges", COLORS["lngn"], "after DINOv2 final LN (fed to adapter)"),
    ):
        e = np.array(h[edges])
        axes[0].stairs(np.array(h[key]) / sum(h[key]), e, color=color, fill=True, alpha=0.5, label=label)
    axes[0].set_xlabel("per-token L2 norm")
    axes[0].set_ylabel("fraction of tokens")
    axes[0].set_title("DINOv2 ViT-S/14 tokens: no high-norm outliers")
    axes[0].legend(frameon=False, fontsize=8)
    ratios = t["init_ctx_over_p5_rms"]
    arms = [a for a in ALL_ARMS if a in ratios]
    axes[1].bar([SHORT[a] for a in arms], [ratios[a] for a in arms], color=[COLORS[a] for a in arms])
    for i, a in enumerate(arms):
        axes[1].text(i, ratios[a] * 1.08, f"{ratios[a]:.3f}", ha="center", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylim(min(ratios.values()) * 0.6, max(ratios.values()) * 2.0)
    axes[1].set_ylabel("RMS(context) / RMS(YOLO P5)")
    axes[1].set_title("Injected scale at initialization")
    return _save(fig, "token_stats")


def resolution_summary(df: pd.DataFrame, bench: dict | None, table: dict) -> tuple[list[dict], str | None]:
    points = []
    rows = {(r["model"], r["arm"], r["dino_res"]): r for r in (bench or {}).get("rows", [])}

    def add(label, sub, key, color):
        if sub.empty:
            return
        b = rows.get(key, {})
        points.append(
            {
                "label": label,
                "n": int(len(sub)),
                "test_map_mean": float(sub["test_map"].mean()),
                "test_map_std": float(sub["test_map"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "gflops": b.get("gflops_total"),
                "latency_ms": b.get("latency_ms"),
                "ms_per_image_batched": b.get("ms_per_image_batched"),
                "color": color,
            }
        )

    voc = df[df["dataset"] == "voc"]
    add("YOLOv8n", voc[(voc["model"] == "yolov8n") & (voc["arm"] == "baseline")], ("yolov8n", "baseline", None), COLORS["baseline"])
    add("YOLOv8s", voc[(voc["model"] == "yolov8s") & (voc["arm"] == "baseline")], ("yolov8s", "baseline", None), COLORS["yolov8s"])
    for res in (224, 336, 448, DEFAULT_RES):
        sub = voc[(voc["model"] == "yolov8n") & (voc["arm"] == "lngn") & (voc["dino_res"] == res)]
        add(f"LN+GN @{res}", sub, ("yolov8n", "lngn", res), COLORS["lngn"])
    plotted = [p for p in points if p["gflops"] is not None]
    if len(plotted) < 2:
        return points, None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, xlabel, ticks in (
        (axes[0], "gflops", "GFLOPs per 640² image (incl. DINOv2)", (8, 10, 20, 30, 50, 100)),
        (axes[1], "ms_per_image_batched", "GPU ms / image (batch 32, FP16, RTX A3000)", (1.5, 2, 3, 5, 10)),
    ):
        for p in plotted:
            ax.errorbar(p[key], 100 * p["test_map_mean"], yerr=100 * p["test_map_std"], fmt="o", color=p["color"], capsize=3)
            text = p["label"] if p["label"] in ("YOLOv8n", "YOLOv8s", "LN+GN @224") else p["label"].replace("LN+GN ", "")
            offset, ha = {"@336": ((-5, 4), "right"), "@448": ((0, -15), "center")}.get(text, ((5, 4), "left"))  # neighbours sit close on the log axis
            ax.annotate(text + (" px" if "@" in text else ""), (p[key], 100 * p["test_map_mean"]), textcoords="offset points", xytext=offset, ha=ha, fontsize=8)
        lngn = sorted([p for p in plotted if p["label"].startswith("LN+GN")], key=lambda p: p[key])
        if len(lngn) > 1:
            ax.plot([p[key] for p in lngn], [100 * p["test_map_mean"] for p in lngn], color=COLORS["lngn"], lw=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test mAP50-95 (%)")
    axes[0].set_title("Accuracy vs compute")
    axes[1].set_title("Accuracy vs measured cost")
    return points, _save(fig, "cost_curve")


def pair_summary(df: pd.DataFrame, dataset: str, label: str) -> tuple[dict, pd.DataFrame]:
    sub = df[(df["dataset"] == dataset) & (df["model"] == "yolov8n") & (df["dino_res"] == DEFAULT_RES)]
    sub = sub[sub["arm"].isin(["baseline", "lngn"])]
    return ({"label": label, **_arm_table(sub)} if not sub.empty else {}), sub


def fig_data_regimes(regimes: list[tuple[str, dict]]) -> str | None:
    regimes = [(n, t) for n, t in regimes if "baseline" in t and "lngn" in t]
    if not regimes:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    width = 0.36
    for i, (name, t) in enumerate(regimes):
        for j, arm in enumerate(("baseline", "lngn")):
            m, s = 100 * t[arm]["test_map_mean"], 100 * (t[arm]["test_map_std"] or 0)
            ax.bar(i + (j - 0.5) * width, m, width, yerr=s, color=COLORS[arm], capsize=3, label=SHORT[arm] if i == 0 else None)
        d = 100 * (t["lngn"]["test_map_mean"] - t["baseline"]["test_map_mean"])
        top = max(100 * t[a]["test_map_mean"] for a in ("baseline", "lngn"))
        ax.text(i, top + 1.2, f"Δ {d:+.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(regimes)), [n for n, _ in regimes])
    ax.set_ylabel("test mAP50-95 (%)")
    ax.set_title("Does the gain depend on labels / domain?")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "data_regimes")


def _seeds(df: pd.DataFrame, group: str, arm: str, model: str = "yolov8n", res: int = DEFAULT_RES) -> dict:
    rows = df[(df["group"] == group) & (df["arm"] == arm) & (df["model"] == model) & (df["dino_res"] == res)]
    return {r.seed: r.records for r in rows.itertuples() if Path(r.records).exists()}


def _family_tests(out: dict) -> None:
    """Holm-adjusted p-values for the normalization family and a TOST-style equivalence verdict for every delta."""
    family = sorted(out.get("main_vs_nonorm", {}).values(), key=lambda d: d["p_two_sided_map"])
    running = 0.0
    for i, d in enumerate(family):
        running = max(running, min(1.0, (len(family) - i) * d["p_two_sided_map"]))
        d["p_holm_map"] = running
    for section in out.values():
        for d in section.values():
            d["equivalent"] = bool(-EQUIV_MARGIN < d["ci90_map"][0] and d["ci90_map"][1] < EQUIV_MARGIN)


def run_bootstrap(df: pd.DataFrame, n_boot: int) -> dict:
    """Seed-paired deltas; all runs scored on the same test images share one set of bootstrap replicates."""
    out: dict = {}

    def compare(sub: pd.DataFrame, comparisons: list[tuple[str, str, dict, dict]]) -> None:
        paths = [p for p in sub["records"] if Path(p).exists()]
        if not paths:
            return
        boot = bootstrap_maps(paths, n_boot=n_boot)
        for section, key, ref, new in comparisons:
            pairs = [(ref[s], new[s]) for s in sorted(ref) if s in new]
            if pairs:
                out.setdefault(section, {})[key] = paired_delta(boot, pairs)

    voc = df[df["dataset"].isin(["voc", "voc10"])]  # both are tested on VOC07 test
    base, nonorm, lngn, gated = (_seeds(voc, "main", a) for a in ("baseline", "nonorm", "lngn", "gated"))
    v8s = _seeds(voc, "capacity", "baseline", model="yolov8s")
    global_, gatedlow = _seeds(voc, "control", "global"), _seeds(voc, "control", "gatedlow")
    comparisons = [("main", a, base, _seeds(voc, "main", a)) for a in FUSION_ARMS]
    comparisons += [("main_vs_nonorm", a, nonorm, _seeds(voc, "main", a)) for a in ("ln", "gn", "lngn", "gated")]
    comparisons += [
        ("lowdata", "lngn", _seeds(voc, "lowdata", "baseline"), _seeds(voc, "lowdata", "lngn")),
        ("control", "global_vs_baseline", base, global_),
        ("control", "spatial_vs_global", global_, lngn),
        ("control", "gatedlow_vs_nonorm", nonorm, gatedlow),
        ("control", "gated_vs_gatedlow", gatedlow, gated),
        ("capacity", "yolov8s_vs_yolov8n", base, v8s),
        ("capacity", "lngn518_vs_yolov8s", v8s, lngn),
    ]
    for res in (224, 336, 448):
        sweep = _seeds(voc, "resolution", "lngn", res=res)
        comparisons += [("resolution", str(res), base, sweep), ("capacity", f"lngn{res}_vs_yolov8s", v8s, sweep)]
    compare(voc, comparisons)

    vis = df[df["dataset"] == "visdrone"]
    vb, vl, vn = (_seeds(vis, "visdrone", a) for a in ("baseline", "lngn", "nonorm"))
    compare(vis, [("visdrone", "lngn", vb, vl), ("visdrone", "nonorm", vb, vn), ("visdrone_vs_nonorm", "lngn", vn, vl)])
    _family_tests(out)
    return out


def error_summary(df: pd.DataFrame) -> tuple[dict, str | None]:
    voc = df[(df["dataset"] == "voc") & (df["model"] == "yolov8n") & (df["dino_res"] == DEFAULT_RES)]
    out = {}
    for arm in (*ALL_ARMS, *CONTROL_ARMS):
        rows = [decompose(p) for p in voc[voc["arm"] == arm]["records"] if Path(p).exists()]
        if not rows:
            continue
        dap = pd.DataFrame([r["dap50"] for r in rows])
        out[arm] = {
            "n": len(rows),
            "ap50_mean": float(np.mean([r["ap50"] for r in rows])),
            "dap50_mean": dap.mean().to_dict(),
            "dap50_std": dap.std(ddof=1).fillna(0.0).to_dict(),
            "counts_mean": pd.DataFrame([r["counts_at_conf"] for r in rows]).mean().to_dict(),
        }
    shown = [a for a in ("baseline", "global", "lngn") if a in out]
    if len(shown) < 2:
        return out, None
    names = {"cls": "wrong class", "loc": "poor box", "both": "class + box", "dupe": "duplicate", "bkg": "background", "miss": "missed"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    width = 0.8 / len(shown)
    for j, arm in enumerate(shown):
        offset = (j - (len(shown) - 1) / 2) * width
        mean = [100 * out[arm]["dap50_mean"][e] for e in ERRORS]
        std = [100 * out[arm]["dap50_std"][e] for e in ERRORS]
        axes[0].barh(np.arange(len(ERRORS)) + offset, mean, width, xerr=std, color=COLORS[arm], label=SHORT[arm], capsize=2)
        counts = out[arm]["counts_mean"]
        kinds = ("cls", "loc", "bkg", "missed")
        axes[1].bar(np.arange(len(kinds)) + offset, [counts[k] for k in kinds], width, color=COLORS[arm], label=SHORT[arm])
    axes[0].set_yticks(range(len(ERRORS)), [names[e] for e in ERRORS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("AP50 points recovered by an oracle fix (dAP)")
    axes[0].set_title("Which errors cost AP? (TIDE-style, VOC07 test)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(range(4), ["wrong class", "poor box", "background", "missed"])
    axes[1].set_ylabel("detections / objects (mean over seeds)")
    axes[1].set_title(f"Error counts at conf ≥ {OPERATING_CONF}")
    return out, _save(fig, "error_types")


def fig_controls(df: pd.DataFrame, table: dict) -> tuple[dict, str | None]:
    control = _arm_table(df[(df["group"] == "control") & (df["dataset"] == "voc")])
    rows = [(a, control if a in CONTROL_ARMS else table) for a in ("baseline", "global", "lngn", "nonorm", "gatedlow", "gated")]
    rows = [(a, t[a]) for a, t in rows if a in t]
    if not control or len(rows) < 2:
        return control, None
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, (arm, t) in enumerate(rows):
        mean, std = 100 * t["test_map_mean"], 100 * (t["test_map_std"] or 0)
        ax.bar(i, mean, 0.65, color=COLORS[arm], alpha=0.9)
        ax.errorbar(i, mean, yerr=std, color="black", capsize=4, lw=1)
        ax.text(i, mean + std + 0.3, f"{mean:.1f}\n(n={t['n']})", ha="center", va="bottom", fontsize=8)
    values = [100 * t["test_map_mean"] for _, t in rows]
    ax.set_ylim(min(values) - 5, max(values) + 3)
    ax.set_xticks(range(len(rows)), [SHORT[a] for a, _ in rows], fontsize=9)
    ax.set_ylabel("VOC07 test mAP50-95 (%)")
    ax.set_title("Controls: spatial layout vs global vector, gate vs init scale")
    return control, _save(fig, "controls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-boot", type=int, default=400)
    args = parser.parse_args()

    df = load_results()
    df = df[df["group"] != "smoke"] if not df.empty else df
    bench_path = RESULTS_ROOT / "bench.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    tokens_path = RESULTS_ROOT / "token_stats.json"
    summary_path = RESULTS_ROOT / "summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": int(len(df)),
        "figures": {},
        "bench": bench,
        "tokens": {k: v for k, v in json.loads(tokens_path.read_text()).items() if k != "hist"} if tokens_path.exists() else None,
    }
    summary["figures"]["token_stats"] = fig_tokens()
    if df.empty:
        summary_path.write_text(json.dumps(summary, indent=2))
        print("no finished runs yet")
        return

    main_df = df[(df["group"] == "main") & (df["dataset"] == "voc") & (df["model"] == "yolov8n")]
    table = _arm_table(main_df)
    summary["main"] = table
    summary["figures"]["main_map"] = fig_main(main_df, table)
    summary["figures"]["dev_curves"] = fig_dev_curves(main_df)
    summary["figures"]["size_bins"] = fig_size_bins(table)
    summary["diagnostics"], summary["figures"]["diagnostics"] = diagnostics_summary(main_df)
    summary["per_class_delta"], summary["figures"]["per_class_delta"] = fig_per_class(main_df)
    summary["cost_curve"], summary["figures"]["cost_curve"] = resolution_summary(df, bench, table)
    full_t, _ = pair_summary(df[df["group"] == "main"], "voc", "VOC 100% labels")
    low_t, _ = pair_summary(df, "voc10", "VOC 10% labels")
    vis_t, _ = pair_summary(df, "visdrone", "VisDrone (aerial)")
    summary["lowdata"], summary["visdrone"] = low_t, vis_t
    summary["visdrone_all"] = _arm_table(df[df["dataset"] == "visdrone"])
    summary["capacity"] = _arm_table(df[df["group"] == "capacity"])
    summary["control"], summary["figures"]["controls"] = fig_controls(df, table)
    summary["errors"], summary["figures"]["error_types"] = error_summary(df)
    summary["equivalence_margin"] = EQUIV_MARGIN
    summary["figures"]["data_regimes"] = fig_data_regimes(
        [("VOC 100%", full_t), ("VOC 10%", low_t), ("VisDrone", vis_t)]
    )
    key = sorted(r for r in df["records"] if Path(r).exists())
    seeds_per_arm = main_df.groupby("arm")["seed"].nunique()
    milestone = len(seeds_per_arm) > 1 and (seeds_per_arm == 3).all()
    if not args.quick:
        summary["bootstrap"] = run_bootstrap(df, args.n_boot)
        summary["bootstrap_generated"], summary["bootstrap_key"] = time.strftime("%Y-%m-%d %H:%M:%S"), key
    elif milestone and previous.get("bootstrap_key") != key:  # every headline arm just reached 3 seeds
        summary["bootstrap"] = run_bootstrap(df, 300)
        summary["bootstrap_generated"], summary["bootstrap_key"] = time.strftime("%Y-%m-%d %H:%M:%S"), key
    else:
        for k in ("bootstrap", "bootstrap_generated", "bootstrap_key"):
            summary[k] = previous.get(k, {} if k == "bootstrap" else None)
    summary["figures"]["delta_ci"] = fig_delta_ci(summary["bootstrap"].get("main", {}))
    summary["qualitative"] = qualitative.build()
    summary["runs"] = _records(df)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["dev_curve", "per_class_map"]).to_csv(RESULTS_ROOT / "runs.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    for arm, t in table.items():
        print(f"{arm:9s} n={t['n']} test mAP50-95 {100 * t['test_map_mean']:.2f} ± {100 * (t['test_map_std'] or 0):.2f}")


if __name__ == "__main__":
    main()
