"""TIDE-style error decomposition of AP50 (Bolya et al., ECCV 2020) from the saved per-image test records.

Every unmatched detection gets one error type, tested in TIDE's order (background, localization, classification,
duplicate, class+localization). Each type is then fixed by an oracle in isolation; the AP50 gained is its dAP.
Matching mirrors Ultralytics ``match_predictions`` at IoU 0.5, so the unfixed AP50 reproduces the reported one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .stats import _EPS, _ap_from_curves

FG, BG = 0.5, 0.1
OPERATING_CONF = 0.25
ERRORS = ("cls", "loc", "both", "dupe", "bkg", "miss")
TP, CLS, LOC, BOTH, DUPE, BKG = range(6)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _match(iou_same: np.ndarray) -> np.ndarray:
    """(gt, pred) index pairs, same procedure as Ultralytics at a single IoU threshold."""
    m = np.argwhere(iou_same >= FG)
    if len(m) > 1:
        m = m[iou_same[m[:, 0], m[:, 1]].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    return m


def _map50(conf: np.ndarray, tp: np.ndarray, cls: np.ndarray, n_gt: np.ndarray, classes: np.ndarray) -> float:
    aps = []
    for c in classes:
        sel = cls == c
        t = tp[sel][np.argsort(-conf[sel])].astype(np.float64)
        if not len(t):
            aps.append(0.0)
            continue
        tpc, fpc = t.cumsum(), (1.0 - t).cumsum()
        aps.append(_ap_from_curves(tpc / (n_gt[c] + _EPS), tpc / (tpc + fpc)))
    return float(np.mean(aps))


def decompose(records_npz: str | Path) -> dict:
    d = np.load(records_npz)
    po, go = d["conf_offsets"], d["target_cls_offsets"]
    conf, pcls, tcls = d["conf"], d["pred_cls"].astype(int), d["target_cls"].astype(int)
    pbox, tbox = d["pred_boxes"].reshape(-1, 4), d["target_boxes"].reshape(-1, 4)
    err = np.full(len(conf), BKG, np.int8)
    target = np.full(len(conf), -1, np.int64)  # GT blamed for a localization / classification error
    match_conf = np.zeros(len(tcls))  # confidence of the detection matched to each GT (0 = unmatched)
    for i in range(len(d["im"])):
        ps, pe, gs, ge = po[i], po[i + 1], go[i], go[i + 1]
        if pe == ps or ge == gs:
            continue
        iou = _iou(tbox[gs:ge], pbox[ps:pe])
        same = tcls[gs:ge, None] == pcls[None, ps:pe]
        iou_same, iou_other = np.where(same, iou, 0.0), np.where(same, 0.0, iou)
        m = _match(iou_same)
        s_max, o_max = iou_same.max(0), iou_other.max(0)
        loc = (iou.max(0) > BG) & (s_max >= BG) & (s_max < FG)
        cls = ~loc & (o_max >= FG)
        dupe = ~loc & ~cls & (s_max >= FG)
        both = ~loc & ~cls & ~dupe & (o_max > BG)
        e = np.full(pe - ps, BKG, np.int8)
        e[both], e[dupe], e[cls], e[loc] = BOTH, DUPE, CLS, LOC
        e[m[:, 1]] = TP
        err[ps:pe] = e
        t = np.full(pe - ps, -1, np.int64)
        t[e == LOC] = gs + iou_same.argmax(0)[e == LOC]
        t[e == CLS] = gs + iou_other.argmax(0)[e == CLS]
        target[ps:pe] = t
        match_conf[gs + m[:, 0]] = conf[ps + m[:, 1]]

    classes = np.unique(tcls)
    n_gt = np.bincount(tcls, minlength=int(max(tcls.max(), pcls.max())) + 1).astype(np.float64)
    used = match_conf > 0
    explained = np.zeros(len(tcls), bool)
    explained[target[(err == LOC) | (err == CLS)]] = True
    missed = ~used & ~explained
    is_tp = err == TP
    base = _map50(conf, is_tp, pcls, n_gt, classes)

    def fix_retarget(kind: int) -> float:
        """Loc/cls oracle: the detection becomes the TP of its GT if that GT is still free, otherwise it is removed."""
        keep, tp, cls = err != kind, is_tp.copy(), pcls.copy()
        claimed = used.copy()
        for j in np.flatnonzero(err == kind)[np.argsort(-conf[err == kind])]:
            g = target[j]
            if not claimed[g]:
                claimed[g], keep[j], tp[j], cls[j] = True, True, True, tcls[g]
        return _map50(conf[keep], tp[keep], cls[keep], n_gt, classes)

    def fix_drop(kind: int) -> float:
        keep = err != kind
        return _map50(conf[keep], is_tp[keep], pcls[keep], n_gt, classes)

    fixed = {
        "cls": fix_retarget(CLS),
        "loc": fix_retarget(LOC),
        "both": fix_drop(BOTH),
        "dupe": fix_drop(DUPE),
        "bkg": fix_drop(BKG),
        "miss": _map50(conf, is_tp, pcls, n_gt - np.bincount(tcls[missed], minlength=len(n_gt)), classes),
    }
    confident = conf >= OPERATING_CONF
    names = {CLS: "cls", LOC: "loc", BOTH: "both", DUPE: "dupe", BKG: "bkg"}
    return {
        "ap50": base,
        "dap50": {k: v - base for k, v in fixed.items()},
        "counts_at_conf": {
            "conf": OPERATING_CONF,
            "tp": int((is_tp & confident).sum()),
            **{name: int(((err == code) & confident).sum()) for code, name in names.items()},
            "missed": int((match_conf < OPERATING_CONF).sum()),
            "n_gt": int(len(tcls)),
        },
    }
