"""Build the final presentation from results/summary.json and results/figures (always buildable; missing
results render as clearly-marked pending panels).

    python -m tools.make_slides
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from nafuse import PROJECT_ROOT, RESULTS_ROOT

OUT = PROJECT_ROOT / "deliverables" / "Final_Presentation_NormFusion.pptx"
FONT = "Segoe UI"
INK = RGBColor(0x15, 0x2E, 0x2A)
TEAL = RGBColor(0x00, 0x7B, 0x73)
CORAL = RGBColor(0xE8, 0x5E, 0x48)
LIME = RGBColor(0xBD, 0xD5, 0x6E)
MINT = RGBColor(0xDC, 0xEC, 0xE8)
SAND = RGBColor(0xF8, 0xE2, 0xDC)
GREY = RGBColor(0x5F, 0x6B, 0x69)
LIGHT = RGBColor(0xF3, 0xF6, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SHORT = {
    "baseline": "YOLOv8n",
    "nonorm": "No norm",
    "ln": "LN only",
    "gn": "GN only",
    "lngn": "LN+GN",
    "gated": "Gated",
    "global": "Global DINO",
    "gatedlow": "Gated, low init",
}
ARMS = ("baseline", "nonorm", "ln", "gn", "lngn", "gated")
TOTAL = 0  # set from the builder list in main()


# ----------------------------------------------------------------------------------------------- formatting
def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}"


def signed(x, digits=1):
    return "–" if x is None else f"{100 * x:+.{digits}f}"


def ci_text(d):
    return f"95% CI [{signed(d['ci_map'][0], 2)}, {signed(d['ci_map'][1], 2)}]"


def excludes_zero(d):
    return d["ci_map"][0] > 0 or d["ci_map"][1] < 0


def ci_short(d):
    return f"{signed(d['delta_map'])} [{signed(d['ci_map'][0])}, {signed(d['ci_map'][1])}]"


def p_text(d, key="p_holm_map"):
    p = d.get(key)
    if p is None:
        return "–"
    return f"< {1 / d['n_boot']:.3f}" if p == 0 else f"= {p:.3f}"


# ----------------------------------------------------------------------------------------------- primitives
def _font(run, size, bold=False, color=INK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def text(slide, x, y, w, h, content, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """``content`` is a string or a list of paragraphs; a paragraph is a string or a list of (text, style) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    paragraphs = content if isinstance(content, list) else [content]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = para if isinstance(para, list) else [(para, {})]
        for chunk, style in runs:
            r = p.add_run()
            r.text = chunk
            _font(r, style.get("size", size), style.get("bold", bold), style.get("color", color), style.get("italic", False))
    return box


def bullets(slide, x, y, w, h, items, size=13, color=INK, gap=6, marker="•"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        runs = item if isinstance(item, list) else [(item, {})]
        r = p.add_run()
        r.text = f"{marker} "
        _font(r, size, True, TEAL)
        for chunk, style in runs:
            r = p.add_run()
            r.text = chunk
            _font(r, style.get("size", size), style.get("bold", False), style.get("color", color))
    return box


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def label_box(slide, x, y, w, h, title, body, fill=MINT, title_color=TEAL, body_size=12, title_size=10):
    rect(slide, x, y, w, h, fill)
    text(slide, x + 0.15, y + 0.1, w - 0.3, 0.3, title.upper(), size=title_size, bold=True, color=title_color)
    text(slide, x + 0.15, y + 0.38, w - 0.3, h - 0.45, body, size=body_size)


def callout(slide, x, y, w, h, big, small, fill=INK, fg=WHITE, accent=LIME):
    rect(slide, x, y, w, h, fill)
    text(slide, x + 0.2, y + 0.12, w - 0.4, h * 0.55, big, size=30, bold=True, color=accent, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + 0.2, y + h * 0.58, w - 0.4, h * 0.4, small, size=11, color=fg)


def arrow(slide, x1, y1, x2, y2, color=GREY, width=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def picture(slide, path, x, y, w, h):
    if not path or not Path(path).exists():
        return pending(slide, x, y, w, h, "figure pending — generated automatically when its runs finish")
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def pending(slide, x, y, w, h, msg):
    s = rect(slide, x, y, w, h, LIGHT, line=RGBColor(0xC9, 0xD3, 0xD1))
    text(slide, x + 0.3, y, w - 0.6, h, msg, size=12, color=GREY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return s


def table(slide, x, y, w, rows, col_w, size=11, header_fill=INK, row_h=0.32, highlight_rows=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            elif i in highlight_rows:
                cell.fill.fore_color.rgb = MINT
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(value)
            _font(r, size, bold=(i == 0 or i in highlight_rows), color=WHITE if i == 0 else INK)
    return tbl


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body.strip()


def new_slide(prs, section, title, lede=None):
    number = len(prs.slides) + 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    text(slide, 0.6, 0.32, 9, 0.3, f"{number:02d}  /  {section.upper()}", size=10, bold=True, color=TEAL)
    text(slide, 0.6, 0.58, 12.1, 0.8, title, size=26 if len(title) <= 60 else 22, bold=True)
    rect(slide, 0.62, 1.32, 0.7, 0.06, CORAL, shape=MSO_SHAPE.RECTANGLE)
    if lede:
        text(slide, 0.6, 1.45, 12.1, 0.5, lede, size=13, color=GREY)
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.6), Inches(7.05), Inches(12.73), Inches(7.05))
    line.line.color.rgb = RGBColor(0xD5, 0xDD, 0xDB)
    line.line.width = Pt(0.75)
    text(slide, 0.6, 7.1, 7, 0.25, "NORMALIZATION-AWARE DINOv2 → YOLOv8 FUSION", size=8, bold=True, color=GREY)
    text(slide, 9.2, 7.1, 3.53, 0.25, f"DLVA FINAL PROJECT  ·  {number}/{TOTAL}", size=8, bold=True, color=GREY, align=PP_ALIGN.RIGHT)
    return slide


# ----------------------------------------------------------------------------------------------- content
class Facts:
    """Read-only view over summary.json with sentence helpers that degrade gracefully."""

    def __init__(self, summary: dict):
        self.s = summary
        self.main = summary.get("main") or {}
        self.boot = (summary.get("bootstrap") or {}).get("main", {})
        self.boot_nn = (summary.get("bootstrap") or {}).get("main_vs_nonorm", {})
        self.bench = {(r["model"], r["arm"], r["dino_res"]): r for r in ((summary.get("bench") or {}).get("rows") or [])}
        self.tokens = summary.get("tokens") or {}
        self.diag = summary.get("diagnostics") or {}
        self.figs = summary.get("figures") or {}
        self.bootall = summary.get("bootstrap") or {}
        self.margin = summary.get("equivalence_margin") or 0.005

    def b(self, section, key):
        return (self.bootall.get(section) or {}).get(key)

    def equivalence_sentence(self):
        arms = [a for a in ("ln", "gn", "lngn") if "ci90_map" in (self.boot_nn.get(a) or {})]
        if not arms:
            return self.delta_sentence("lngn", "nonorm")
        eq = sum(bool(self.boot_nn[a].get("equivalent")) for a in arms)
        widest = max(abs(v) for a in arms for v in self.boot_nn[a]["ci90_map"])
        out = f"Normalization: {eq} of {len(arms)} LN/GN arms are equivalent to no norm (90% CIs within ±{100 * widest:.2f} < ±{100 * self.margin:.1f} points)"
        g = self.boot_nn.get("gated")
        if g:
            out += f"; gated {signed(g['delta_map'], 2)}, Holm p {p_text(g)}"
        return out

    def h1_rejected(self):
        arms = [self.boot_nn.get(a) or {} for a in ("ln", "gn", "lngn")]
        return all(d.get("equivalent") for d in arms)

    def mean(self, arm, key="test_map"):
        return (self.main.get(arm) or {}).get(f"{key}_mean")

    def std(self, arm, key="test_map"):
        return (self.main.get(arm) or {}).get(f"{key}_std")

    def n(self, arm):
        return (self.main.get(arm) or {}).get("n", 0)

    def delta(self, arm, ref="baseline"):
        a, b = self.mean(arm), self.mean(ref)
        return None if a is None or b is None else a - b

    def delta_sentence(self, arm, ref="baseline"):
        d = self.delta(arm, ref)
        if d is None:
            return f"{SHORT[arm]} vs {SHORT[ref]}: pending"
        boot = (self.boot if ref == "baseline" else self.boot_nn if ref == "nonorm" else {}).get(arm)
        n = min(self.n(arm), self.n(ref))
        tail = f"{n} seed{'s' if n != 1 else ''}"
        if boot:
            tail += f", {ci_text(boot)}" + (" — beyond noise" if excludes_zero(boot) else " — within noise")
        return f"{SHORT[arm]} vs {SHORT[ref]}: {signed(d)} mAP50-95 ({tail})"

    def fig(self, name):
        return self.figs.get(name)


def slide_title(prs, f: Facts):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, 13.333, 7.5, WHITE, shape=MSO_SHAPE.RECTANGLE)
    rect(slide, 8.55, 0, 4.783, 7.5, INK, shape=MSO_SHAPE.RECTANGLE)
    text(slide, 0.7, 0.75, 7.5, 0.3, "FINAL PROJECT  ·  DEEP LEARNING FOR VISUAL ANALYTICS", size=11, bold=True, color=TEAL)
    text(slide, 0.7, 1.45, 7.6, 2.0, ["Make spatial", "context count."], size=46, bold=True)
    rect(slide, 0.72, 3.25, 0.9, 0.07, CORAL, shape=MSO_SHAPE.RECTANGLE)
    text(slide, 0.7, 3.45, 7.6, 1.0, "Normalization-aware DINOv2 → YOLOv8 feature fusion: what actually makes a frozen ViT help a pretrained CNN detector, and is it worth the cost?", size=16, color=GREY)
    text(slide, 0.7, 5.6, 7, 0.4, "Shantha Suresh M", size=16, bold=True)
    text(slide, 0.7, 6.0, 7, 0.4, f"Individual project  ·  {time.strftime('%d %B %Y')}", size=12, color=GREY)

    d, boot = f.delta("lngn"), f.boot.get("lngn")
    big = signed(d) if d is not None else "pending"
    small = "mAP50-95 points, LN+GN fusion vs YOLOv8n on held-out VOC07 test"
    if boot:
        small += f"\n{ci_text(boot)}, {f.n('lngn')} seeds"
    callout(slide, 8.95, 0.9, 3.95, 1.7, big, small, fill=RGBColor(0x29, 0x4A, 0x43))
    runs = f.s.get("n_runs", 0)
    hours = sum(r.get("train_hours", 0) or 0 for r in f.s.get("runs", []))
    callout(slide, 8.95, 2.85, 3.95, 1.7, f"{runs}", f"trained & held-out-evaluated runs\n{hours:.0f} GPU-hours of training", fill=RGBColor(0x29, 0x4A, 0x43))
    b = f.bench.get(("yolov8n", "lngn", 518), {})
    base = f.bench.get(("yolov8n", "baseline", None), {})
    cost = f"{b['gflops_total'] / base['gflops_total']:.0f}×" if b and base else "pending"
    callout(slide, 8.95, 4.8, 3.95, 1.7, cost, "compute of the fused model vs YOLOv8n (GFLOPs at 640², incl. DINOv2 @518)", fill=RGBColor(0x29, 0x4A, 0x43))
    notes(slide, """
Opening line: "Last time I showed a +7.5 mAP jump from fusing DINOv2 into YOLOv8n — from one run where five things changed at once.
Today I'll show what actually causes the gain, how big it is once you control for noise, and what it costs."
""")


def slide_problem(prs, f: Facts):
    s = new_slide(prs, "Starting point", "Midterm evidence: a big jump — but five things changed at once", "Single-seed IUI runs (10 epochs, batch 2). The test split was also used to pick the checkpoint.")
    rows = [
        ["Model (historical, 1 seed)", "mAP50", "mAP50-95", "ms/img"],
        ["YOLOv8n baseline", "68.5", "47.7", "2.7"],
        ["+ DINOv2 global token", "36.9", "21.8", "36.9"],
        ["+ spatial tokens, gated", "82.1", "55.2", "34.8"],
        ["+ spatial tokens, LN+GN", "81.6", "54.1", "33.1"],
    ]
    table(s, 0.6, 2.15, 6.3, rows, [3.0, 1.1, 1.1, 1.1], size=12, row_h=0.42)
    label_box(
        s, 7.3, 2.15, 5.45, 3.9, "What that evidence could not separate",
        [
            "• Token type, gate, normalization and initialization all changed together",
            "• One seed: no idea how large run-to-run noise is",
            "• Checkpoint chosen on the same split it was scored on",
            [("• A weight-transfer bug: ", {"bold": True}), ("DINO channels were concatenated first, but pretrained filters were copied as if appended — so the YOLO neck read shifted channels.", {})],
            "• Latency compared framework timings, not matched end-to-end cost",
        ],
        fill=SAND, title_color=CORAL, body_size=12,
    )
    text(s, 0.6, 4.45, 6.3, 1.6, [[("Hypothesis, not a result. ", {"bold": True, "color": CORAL}), ("This project re-runs the comparison so that each factor is isolated, every claim has seeds and confidence intervals, and cost is measured on the same hardware.", {})]], size=13)
    notes(s, """
Be upfront: the midterm numbers motivated the study; they don't establish the cause. Mention the channel-offset bug explicitly —
finding and fixing it is part of the contribution, and it's why the old 'baseline vs fusion' gap could not be trusted.
""")


def slide_question(prs, f: Facts):
    s = new_slide(prs, "Research question", "Does normalization make DINOv2 context reliably useful?")
    rect(s, 0.6, 1.75, 12.15, 1.15, INK)
    text(s, 0.85, 1.83, 11.7, 1.0, "Q: When frozen DINOv2 patch features are fused into a pretrained YOLOv8n, does explicit normalization (token LayerNorm, projected GroupNorm) — rather than fusion itself — explain a stable gain, and does that gain justify the ViT's inference cost?", size=15, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    hyps = [
        ("H1  Normalization", "Arms with LN and/or GN beat the no-norm arm beyond seed noise."),
        ("H2  Fusion", "The supplied LN+GN design beats a matched YOLOv8n baseline on untouched test data."),
        ("H3  Cost", "The gain survives cost-matched comparisons: a bigger CNN (YOLOv8s) and cheaper DINOv2 resolutions."),
        ("H4  Regime", "Gains grow with fewer labels and shrink on a far domain (aerial VisDrone)."),
    ]
    for i, (h, body) in enumerate(hyps):
        x = 0.6 + i * 3.08
        label_box(s, x, 3.15, 2.9, 2.15, h, body, fill=MINT if i % 2 == 0 else SAND, title_color=TEAL if i % 2 == 0 else CORAL, body_size=12)
    text(s, 0.6, 5.55, 12.1, 1.3, [[("What would count as a negative result: ", {"bold": True}), ("no arm separates from no-norm (H1 rejected), or the fusion gain is inside the 95% CI of zero, or a 3.5× cheaper CNN gives the same accuracy. Each is reported as-is.", {})]], size=13, color=GREY)
    notes(s, "Read the question once. Then: 'four hypotheses, each with a pre-declared way to fail'.")


def slide_related(prs, f: Facts):
    s = new_slide(prs, "Related work & gap", "ViT→CNN fusion is common; isolating why it works is not")
    rows = [
        ["Work", "What it establishes", "What it leaves open here"],
        ["DINOv2 (Oquab et al., 2023)", "Frozen SSL patch features transfer to dense tasks", "How to inject them into a pretrained CNN"],
        ["ViTDet (Li et al., ECCV 2022)", "Plain ViT backbones work for detection", "Uses the ViT as the backbone, not as side context"],
        ["ViT-Adapter (Chen et al., ICLR 2023)", "Adapters + normalization bridge ViT and dense heads", "No isolation of individual norm layers"],
        ["Registers (Darcet et al., ICLR 2024)", "Large DINOv2 models emit high-norm artifact tokens", "Whether small ViT-S/14 needs token norm at all"],
        ["GroupNorm (Wu & He, ECCV 2018)", "Batch-size-independent normalization", "Its role when two feature streams are concatenated"],
        ["Zero-init / ControlNet (Zhang et al., ICCV 2023)", "Zero-initialized injections preserve a pretrained model", "Combined with channel-correct weight transfer"],
    ]
    table(s, 0.6, 1.8, 12.15, rows, [3.6, 4.45, 4.1], size=11, row_h=0.47)
    rect(s, 0.6, 5.25, 12.15, 1.55, MINT)
    text(s, 0.85, 5.35, 11.7, 1.4, [[("Gap addressed: ", {"bold": True, "color": TEAL}), ("a controlled, multi-seed ablation of each normalization component in a ViT→CNN-detector fusion, with function-preserving, channel-correct initialization, training-time mechanism measurements, and cost measured end-to-end.", {})]], size=14, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Keep this short — 40 seconds. The point: nobody isolates the normalization; that is the gap.")


def slide_method(prs, f: Facts):
    s = new_slide(prs, "Method", "Architecture: DINOv2 context enters the neck right after SPPF")
    y = 2.0
    boxes = [
        (0.6, y + 1.05, 1.5, 0.75, "Image\n640×640", LIGHT, INK),
        (2.5, y + 1.75, 2.1, 0.75, "YOLOv8n backbone\n(pretrained)", MINT, INK),
        (5.0, y + 1.75, 1.8, 0.75, "SPPF → P5\n256 × 20 × 20", MINT, INK),
        (2.5, y + 0.25, 2.1, 0.75, "DINOv2 ViT-S/14\nfrozen · 518² input", SAND, INK),
        (5.0, y + 0.25, 1.8, 0.75, "patch tokens\n384 × 37 × 37", SAND, INK),
        (7.2, y + 0.25, 3.3, 0.75, "[LN] → 1×1 conv 384→64 → [GN] → ×α\nresize to 20 × 20", CORAL, WHITE),
        (8.15, y + 1.75, 1.4, 0.75, "concat\n256 + 64", INK, WHITE),
        (10.0, y + 1.75, 2.75, 0.75, "YOLOv8 neck + head\n(widened L12, L21)", MINT, INK),
    ]
    for x, yy, w, h, label, fill, fg in boxes:
        rect(s, x, yy, w, h, fill)
        text(s, x, yy, w, h, label, size=11, bold=True, color=fg, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 2.1, y + 1.42, 2.5, y + 2.1)
    arrow(s, 2.1, y + 1.42, 2.5, y + 0.62)
    arrow(s, 4.6, y + 2.12, 5.0, y + 2.12)
    arrow(s, 4.6, y + 0.62, 5.0, y + 0.62)
    arrow(s, 6.8, y + 0.62, 7.2, y + 0.62)
    arrow(s, 6.8, y + 2.12, 8.15, y + 2.12)
    arrow(s, 8.85, y + 1.0, 8.85, y + 1.75)
    arrow(s, 9.55, y + 2.12, 10.0, y + 2.12)
    label_box(s, 0.6, 4.85, 4.0, 2.0, "Five arms + two controls", "No norm · LN only · GN only · LN+GN (supplied) · gated. α = 0.05; gated: σ(g)·0.1. Controls: LN+GN on the image-mean token (no layout); gated with the small projector init.", body_size=11.5)
    label_box(s, 4.8, 4.85, 3.9, 2.0, "Fix 1 — channel-correct transfer", "P5 keeps its channels first; DINO channels are appended. Pretrained filters of the two consuming convs are copied at their exact offsets (L12: 256|64|128, L21: 128|256|64).", fill=SAND, title_color=CORAL, body_size=12)
    label_box(s, 8.9, 4.85, 3.85, 2.0, "Fix 2 — function-preserving init", "DINO input columns start at zero ⇒ the untrained fused model is the baseline. Unit test: max |Δ output| = 0 for all five arms.", body_size=12)
    notes(s, """
Walk the diagram left to right. Then the two fixes: (1) the old code shifted the pretrained filters; (2) zero-initialized DINO columns
mean every arm starts *exactly* at the baseline, so any difference after training is caused by training dynamics, not by a perturbed start.
DINOv2 is held outside the model's parameters: never trained, never in the checkpoint, never touched by EMA.
""")


def slide_protocol(prs, f: Facts):
    s = new_slide(prs, "Protocol", "Same data, schedule and init for every arm; test used once")
    rows = [
        ["Split", "Images", "Role"],
        ["VOC12 train+val + VOC07 train", "14,041", "training"],
        ["VOC07 val", "2,510", "checkpoint selection (dev)"],
        ["VOC07 test", "4,952", "final evaluation, once per run"],
        ["10% seeded subset of train", "1,404", "low-label regime"],
        ["VisDrone2019 train / val / test-dev", "6,471 / 548 / 1,610", "second domain"],
    ]
    table(s, 0.6, 1.85, 6.5, rows, [3.2, 1.5, 1.8], size=11, row_h=0.42)
    bullets(
        s, 7.4, 1.85, 5.4, 4.9,
        [
            "YOLOv8n from COCO weights; identical class-head initialization",
            "15 epochs (30 at 10% labels), AdamW lr 1e-3, batch 16, 640², AMP, 1 warm-up epoch, mosaic off for last 2",
            "3 seeds per headline arm; seed changes init of new layers and data order",
            "Best epoch by dev fitness only; test evaluated once with the selected checkpoint",
            "Metrics: mAP50-95, mAP50, P, R, COCO small/medium/large AP, params, FLOPs, end-to-end latency incl. DINOv2",
            "Statistics: seed-paired image bootstrap (400 replicates) → 95% CI of Δ mAP; equivalence = 90% CI inside ±0.5 points (TOST); Holm correction over the 4 normalization comparisons",
            "Mechanism logging every 50 iterations: feature-scale ratio, gradient ratio, DINO share of the first neck conv",
            "Error analysis: TIDE-style AP50 decomposition from the saved per-image matches",
        ],
        size=11.5,
        gap=4,
    )
    notes(s, "Stress: dev for selection, test once. The bootstrap resamples test images jointly for paired models, so the CI is about the difference.")


def slide_experiments(prs, f: Facts):
    runs = f.s.get("runs", [])
    hours = sum(r.get("train_hours", 0) or 0 for r in runs)
    s = new_slide(prs, "Experiment map", "Depth: every claim has its own controlled experiment", f"{len(runs)} trained runs, {hours:.0f} GPU-hours. Every run: checkpoint picked on VOC07 val, scored once on the held-out test split.")

    def n(**kw):
        return sum(all(r.get(k) == v for k, v in kw.items()) for r in runs)

    def ans(d, fmt=ci_short):
        return fmt(d) if d else "pending (Kaggle run)"

    pts = {p["label"]: p for p in (f.s.get("cost_curve") or [])}
    res = ", ".join(f"{k[7:]}: {pct(pts[k]['test_map_mean'])}" for k in ("LN+GN @224", "LN+GN @448", "LN+GN @518") if k in pts)
    e = f.s.get("errors") or {}
    err = f"wrong-class dAP {100 * e['baseline']['dap50_mean']['cls']:.1f} → {100 * e['lngn']['dap50_mean']['cls']:.1f}" if "baseline" in e and "lngn" in e else "pending"
    shares = [v["L12/dino_share"]["end"] for v in f.diag.values() if "L12/dino_share" in v]
    mech = f"final DINO share {min(shares):.2f}–{max(shares):.2f} across arms" if shares else "pending"
    norm = "pending"
    if "ci90_map" in (f.boot_nn.get("lngn") or {}):
        norm = "no — LN/GN ≡ no norm (±0.5)" if f.h1_rejected() else "no significant difference"
    rows = [
        ["Question", "Experiment (one factor varies)", "Runs", "Answer (Δ mAP50-95, 95% CI)"],
        ["Does fusion help?", "YOLOv8n vs LN+GN fusion, 3 seeds", n(group="main", arm="baseline") + n(group="main", arm="lngn"), ans(f.boot.get("lngn"))],
        ["H1: does normalization matter?", "No norm / LN / GN / LN+GN / gated adapters", n(group="main") - n(group="main", arm="baseline"), norm],
        ["Is it the spatial layout?", "LN+GN on the image-mean DINO token", n(group="control", arm="global"), ans(f.b("control", "spatial_vs_global"))],
        ["Why does gating win?", "Gated arm with the small projector init", n(group="control", arm="gatedlow"), ans(f.b("control", "gated_vs_gatedlow"))],
        ["H3: is it just compute?", "YOLOv8s baseline (3.5× YOLOv8n FLOPs)", n(group="capacity"), ans(f.b("capacity", "lngn518_vs_yolov8s"))],
        ["How much ViT resolution?", "DINOv2 input 224 / 336 / 448 / 518 px", n(group="resolution"), res or "pending"],
        ["H4: fewer labels?", "10% of VOC labels, baseline vs LN+GN", n(group="lowdata"), ans(f.b("lowdata", "lngn"))],
        ["H4: another domain?", "VisDrone aerial: baseline / no norm / LN+GN", n(group="visdrone"), ans(f.b("visdrone", "lngn"))],
        ["Which errors are fixed?", "TIDE-style AP50 error decomposition", "analysis", err],
        ["How is DINO used in training?", "DINO share, feature & gradient ratios / 50 it", "analysis", mech],
    ]
    table(s, 0.6, 2.0, 12.15, rows, [2.75, 4.5, 0.8, 4.1], size=10.5, row_h=0.43)
    notes(s, "This slide is the answer to 'depth of experimentation': each row is a separate, controlled question with its own runs. Rows marked pending are running on Kaggle.")


def slide_statistics(prs, f: Facts):
    s = new_slide(prs, "Why normalization could matter", "Each normalization arm injects DINOv2 at a different scale")
    picture(s, f.fig("token_stats"), 0.6, 1.8, 8.3, 3.6)
    t = f.tokens
    if t:
        post, ch, ratios = t["dino_token_norm_post_final_norm"], t["dino_channel_std"], t["init_ctx_over_p5_rms"]
        items = [
            [("No outlier tokens: ", {"bold": True}), (f"max/median token norm {post['max_over_median']:.2f} — the 'registers' artifact does not show up in ViT-S/14, so LN is not needed for that.", {})],
            [("Channel anisotropy: ", {"bold": True}), (f"per-channel std spans {ch['max_over_min']:.1f}× after DINOv2's own final LN.", {})],
            [("Scale gap: ", {"bold": True}), (f"YOLO P5 RMS {t['yolo_p5_rms_mean']:.2f} vs DINO token RMS {t['dino_post_norm_rms']:.2f}.", {})],
            [("Injected at init: ", {"bold": True}), (f"GN arms {ratios['lngn']:.3f}×P5, no-norm {ratios['nonorm']:.3f}×, LN-only {ratios['ln']:.3f}× — a {ratios['lngn'] / ratios['ln']:.0f}× spread from normalization alone.", {})],
        ]
    else:
        items = ["token statistics pending"]
    bullets(s, 9.1, 1.8, 3.7, 5.0, items, size=11.5, gap=8)
    text(s, 0.6, 5.65, 8.3, 1.2, [[("Testable prediction: ", {"bold": True, "color": CORAL}), ("normalization decides how loudly the frozen ViT stream speaks relative to the CNN stream — a 10× spread at initialization. If that scale drives the gain, the arms must end at different accuracies. The next slides test exactly this.", {})]], size=12)
    notes(s, "This is the 'why LN+GN' argument turned into a falsifiable prediction. Numbers on the right are measured on 400 dev images.")


def slide_main(prs, f: Facts):
    s = new_slide(prs, "Main result", "Held-out VOC07 test: does fusion help beyond seed noise?")
    picture(s, f.fig("main_map"), 0.6, 1.75, 7.2, 3.7)
    if f.fig("delta_ci"):
        picture(s, f.fig("delta_ci"), 0.6, 5.35, 7.2, 1.6)
    else:
        pending(s, 0.6, 5.45, 7.2, 1.45, "bootstrap CIs are computed when the queue finishes (python -m tools.analyze)")
    rows = [["Arm", "n", "mAP50-95", "mAP50", "Δ vs base"]]
    for arm in ARMS:
        if arm in f.main:
            d = f.delta(arm)
            rows.append([SHORT[arm], f.n(arm), f"{pct(f.mean(arm))} ± {pct(f.std(arm) or 0)}", pct(f.mean(arm, "test_map50")), "—" if arm == "baseline" else signed(d)])
    if len(rows) > 1:
        hl = [i for i, r in enumerate(rows) if r[0] == "LN+GN"]
        table(s, 8.05, 1.8, 4.7, rows, [1.25, 0.45, 1.25, 0.85, 0.9], size=10.5, row_h=0.36, highlight_rows=hl)
    lines = [f.delta_sentence("lngn"), f.equivalence_sentence()]
    bullets(s, 8.05, 1.95 + 0.36 * len(rows), 4.7, 2.6, [l for l in lines if l], size=11)
    notes(s, "Say the headline number with its CI. 'Not significant' is not the claim for normalization — the claim is equivalence: the 90% CI of every LN/GN arm vs no-norm lies inside ±0.5 points.")


def slide_mechanism(prs, f: Facts):
    s = new_slide(prs, "Mechanism", "How much does the detector actually listen to DINOv2?")
    picture(s, f.fig("diagnostics"), 0.6, 1.7, 8.0, 5.2)
    items = []
    for arm in ("nonorm", "ln", "gn", "lngn", "gated"):
        d = f.diag.get(arm, {}).get("L12/dino_share")
        if d:
            items.append([(f"{SHORT[arm]}: ", {"bold": True}), (f"DINO share of first neck conv {d['start']:.2f} → {d['end']:.2f}", {})])
    bullets(s, 8.85, 1.8, 3.95, 3.2, items or ["diagnostics pending"], size=11.5)
    shares = [f.diag[a]["L12/dino_share"]["end"] for a in ("nonorm", "ln", "gn", "lngn") if "L12/dino_share" in f.diag.get(a, {})]
    maps = [f.mean(a) for a in ("nonorm", "ln", "gn", "lngn") if f.mean(a) is not None]
    if len(shares) > 1 and len(maps) > 1:
        text(s, 8.85, 4.55, 3.95, 1.0, [[("Reading: ", {"bold": True, "color": CORAL}), (f"reliance on DINOv2 differs {max(shares) / min(shares):.1f}× between the normalization arms, accuracy by only {100 * (max(maps) - min(maps)):.2f} points — the detector rescales the context itself.", {})]], size=11)
    text(s, 8.85, 5.75, 3.95, 1.2, "DINO share = RMS of the conv response to DINO channels ÷ (DINO + YOLO responses). It starts at exactly 0 (zero-init).", size=10, color=GREY)
    notes(s, "Interpretation: normalization changes how much the detector leans on DINOv2, but every arm reaches the same accuracy — so the scale argument for LN+GN does not survive training.")


def slide_controls(prs, f: Facts):
    s = new_slide(prs, "Controls", "What part of DINOv2 helps, and why does the gated design win?", "Two extra arms, each changing exactly one factor of an existing arm.")
    picture(s, f.fig("controls"), 0.6, 1.95, 7.4, 3.9)
    items = []
    lay, glob = f.b("control", "spatial_vs_global"), f.b("control", "global_vs_baseline")
    if lay and glob:
        total = lay["delta_map"] + glob["delta_map"]
        share = f" — {100 * lay['delta_map'] / total:.0f}% of the fusion gain needs the layout" if total > 0 else ""
        items.append([("Layout vs global vector: ", {"bold": True}), (f"the image-mean DINO token gives {signed(glob['delta_map'])} over YOLOv8n; the spatial map adds {signed(lay['delta_map'])} ({ci_text(lay)}){share}.", {})])
    else:
        items.append([("Layout vs global vector: ", {"bold": True}), ("pending (Kaggle run).", {})])
    gate, low = f.b("control", "gated_vs_gatedlow"), f.b("control", "gatedlow_vs_nonorm")
    if gate and low:
        low_gain = excludes_zero(low) and low["delta_map"] > 0
        init_gain = excludes_zero(gate) and gate["delta_map"] > 0
        cause = "both the gate and the larger init" if low_gain and init_gain else "the learnable gate" if low_gain else "the larger projector init" if init_gain else "neither factor alone (within noise)"
        items.append([("Gate vs init: ", {"bold": True}), (f"gated with the small init vs no norm {signed(low['delta_map'], 2)} ({ci_text(low)}); larger init {signed(gate['delta_map'], 2)} ({ci_text(gate)}) → the gated advantage comes from {cause}.", {})])
    else:
        items.append([("Gate vs init: ", {"bold": True}), ("pending (Kaggle run). The gated arm differs from no norm in two ways — learnable gate and 4× larger projector init — this run separates them.", {})])
    bullets(s, 8.25, 1.95, 4.5, 4.9, items, size=11.5, gap=10)
    text(s, 0.6, 6.0, 7.4, 0.9, "Global control: the same LN+GN adapter, but the 37×37 token map is averaged to one vector and broadcast — same channels, same GN-fixed scale, no spatial layout.", size=10.5, color=GREY)
    notes(s, "If the global vector gets most of the gain, DINOv2 acts as an image-level scene prior; if the layout matters, it is acting as dense, spatially aligned semantics.")


def slide_curves(prs, f: Facts):
    s = new_slide(prs, "Training budget", "Does the gap close as YOLO trains longer?")
    picture(s, f.fig("dev_curves"), 0.6, 1.75, 7.6, 4.6)
    items = []
    if f.mean("baseline") is not None:
        be = (f.main["baseline"].get("best_epoch_mean"))
        items.append(f"Baseline best dev epoch ≈ {be:.0f} of 15" if be else "")
    for arm in ("lngn", "nonorm"):
        if arm in f.main and f.main[arm].get("best_epoch_mean"):
            items.append(f"{SHORT[arm]} best dev epoch ≈ {f.main[arm]['best_epoch_mean']:.0f}")
    items.append("A gain that shrinks every epoch is a 'faster convergence' effect; a stable gap is a capacity effect.")
    bullets(s, 8.5, 1.9, 4.3, 4.5, [i for i in items if i], size=12)
    notes(s, "Use the dev curves (selection split) — not test — to discuss training dynamics.")


def slide_where(prs, f: Facts):
    s = new_slide(prs, "Where it helps", "Object size and class: what the context fixes")
    picture(s, f.fig("size_bins"), 0.6, 1.75, 6.4, 3.4)
    picture(s, f.fig("per_class_delta"), 7.2, 1.75, 5.6, 5.1)
    items = []
    for key, label in (("test_map_small", "small"), ("test_map_medium", "medium"), ("test_map_large", "large")):
        b, l = f.mean("baseline", key), f.mean("lngn", key)
        if b is not None and l is not None:
            items.append(f"{label}: {pct(b)} → {pct(l)} ({signed(l - b)})")
    pc = f.s.get("per_class_delta") or {}
    if pc:
        best = sorted(pc.items(), key=lambda kv: -kv[1])[:3]
        worst = sorted(pc.items(), key=lambda kv: kv[1])[:2]
        items.append("largest gains: " + ", ".join(f"{k} {v:+.1f}" for k, v in best))
        items.append("largest drops: " + ", ".join(f"{k} {v:+.1f}" for k, v in worst))
    bullets(s, 0.6, 5.3, 6.4, 1.6, items or ["size/class breakdown pending"], size=11.5, gap=3)
    notes(s, "DINOv2 tokens are 14-px patches resized to a 20×20 grid: expect help on medium/large objects and semantic confusions, little on tiny objects.")


def slide_errors(prs, f: Facts):
    s = new_slide(prs, "Error analysis", "Which detection errors does DINOv2 context remove?", "TIDE-style decomposition (Bolya et al., ECCV 2020): each error type is fixed by an oracle; the AP50 it recovers is its cost.")
    picture(s, f.fig("error_types"), 0.6, 1.95, 8.4, 3.9)
    e = f.s.get("errors") or {}
    items, change = [], {}
    if "baseline" in e and "lngn" in e:
        b, l = e["baseline"], e["lngn"]
        for key, label in (("cls", "wrong class"), ("loc", "poor box"), ("bkg", "background"), ("miss", "missed")):
            x, y = 100 * b["dap50_mean"][key], 100 * l["dap50_mean"][key]
            change[label] = (y - x) / x if x else 0.0
            items.append([(f"{label}: ", {"bold": True}), (f"{x:.2f} → {y:.2f} AP50 ({100 * change[label]:+.0f}%)", {})])
        cb, cl = b["counts_mean"], l["counts_mean"]
        items.append([("At conf ≥ 0.25: ", {"bold": True}), (f"wrong-class detections {cb['cls']:.0f} → {cl['cls']:.0f}; missed objects {cb['missed']:.0f} → {cl['missed']:.0f} of {cb['n_gt']:.0f}; background detections {cb['bkg']:.0f} → {cl['bkg']:.0f}", {})])
    bullets(s, 9.2, 1.95, 3.6, 4.9, items or ["error analysis pending"], size=11, gap=6)
    if change:
        ranked = sorted(change.items(), key=lambda kv: kv[1])
        text(s, 0.6, 6.0, 8.4, 0.9, [[("Reading: ", {"bold": True, "color": CORAL}), (f"the largest relative drop is '{ranked[0][0]}' ({100 * ranked[0][1]:+.0f}%), the smallest '{ranked[-1][0]}' ({100 * ranked[-1][1]:+.0f}%). Means over 3 seeds per arm.", {})]], size=12)
    notes(s, "Each non-matching detection is assigned one error type in TIDE's order (background, localization, classification, duplicate, both). dAP = AP50 after fixing only that type minus AP50 as measured. The unfixed AP50 reproduces Ultralytics' value to 0.01 points.")


def slide_cost(prs, f: Facts):
    s = new_slide(prs, "Cost", "Is the accuracy worth a ViT? Cost and a bigger-CNN control")
    picture(s, f.fig("cost_curve"), 0.6, 1.75, 8.0, 3.2)
    rows = [["Model", "GFLOPs", "ms/img (b32)", "ms (b1)", "params"]]
    for key, label in (
        (("yolov8n", "baseline", None), "YOLOv8n"),
        (("yolov8s", "baseline", None), "YOLOv8s"),
        (("yolov8n", "lngn", 224), "LN+GN @224"),
        (("yolov8n", "lngn", 336), "LN+GN @336"),
        (("yolov8n", "lngn", 448), "LN+GN @448"),
        (("yolov8n", "lngn", 518), "LN+GN @518"),
    ):
        b = f.bench.get(key)
        if b:
            rows.append([label, f"{b['gflops_total']:.1f}", f"{b['ms_per_image_batched']:.2f}", f"{b['latency_ms']:.1f}", f"{b['params_total'] / 1e6:.1f}M"])
    if len(rows) > 1:
        table(s, 8.85, 1.8, 3.9, rows, [1.25, 0.65, 0.75, 0.6, 0.65], size=9.5, row_h=0.33)
    points = {p["label"]: p for p in (f.s.get("cost_curve") or [])}
    items = []
    if "YOLOv8s" in points and "LN+GN @518" in points:
        v8s, fused = points["YOLOv8s"], points["LN+GN @518"]
        better = fused["test_map_mean"] > v8s["test_map_mean"]
        items.append(f"Bigger CNN control: YOLOv8s {pct(v8s['test_map_mean'])} vs fused {pct(fused['test_map_mean'])} mAP50-95 — " + ("fusion wins on accuracy" if better else "the bigger CNN wins"))
    res = sorted((p for p in points.values() if p["label"].startswith("LN+GN @")), key=lambda p: p["gflops"] or 0)
    if len(res) > 1:
        items.append("DINO resolution sweep: " + ", ".join(f"{p['label'][7:]}: {pct(p['test_map_mean'])}" for p in res))
    items.append("Latency at batch 1 on this laptop GPU is launch-bound; GFLOPs and batched time are the fair cost measures.")
    bullets(s, 0.6, 5.1, 12.1, 1.8, items, size=12, gap=3)
    notes(s, "Lead with the YOLOv8s comparison — it is the strongest objection ('why not just use a bigger YOLO?').")


def slide_regimes(prs, f: Facts):
    s = new_slide(prs, "Generalization", "Fewer labels and a far domain")
    picture(s, f.fig("data_regimes"), 0.6, 1.75, 7.4, 4.0)
    items = []
    for key, label in (("lowdata", "VOC 10% labels"), ("visdrone", "VisDrone (aerial, tiny objects)")):
        t = f.s.get(key) or {}
        if "baseline" in t and "lngn" in t:
            d = t["lngn"]["test_map_mean"] - t["baseline"]["test_map_mean"]
            boot = ((f.s.get("bootstrap") or {}).get(key) or {}).get("lngn")
            tail = f", {ci_text(boot)}" if boot else ""
            items.append(f"{label}: {pct(t['baseline']['test_map_mean'])} → {pct(t['lngn']['test_map_mean'])} ({signed(d)}, n={min(t['baseline']['n'], t['lngn']['n'])}{tail})")
        else:
            items.append(f"{label}: pending")
    d = f.delta("lngn")
    if d is not None:
        items.append(f"Reference, VOC 100% labels: {signed(d)}")
    vn = f.b("visdrone_vs_nonorm", "lngn")
    items.append(f"VisDrone, LN+GN vs no norm: {ci_short(vn)} — " + ("normalization matters here" if excludes_zero(vn) else "normalization again within noise") if vn else "VisDrone, LN+GN vs no norm (H1 on a second domain): pending")
    bullets(s, 8.3, 1.9, 4.5, 4.5, items, size=12)
    text(s, 0.6, 5.95, 12.1, 0.9, "No generalization claim is made from VOC alone; VisDrone tests whether frozen natural-image features still help when objects are tiny and viewpoints are aerial.", size=11.5, color=GREY)
    notes(s, "If the 10%-label gain is larger, the story is 'self-supervised features substitute for labels'.")


def slide_qualitative(prs, f: Facts):
    s = new_slide(prs, "Examples", "Where fusion helps — and where it hurts")
    q = f.s.get("qualitative") or {}
    figs = q.get("figures") or [None, None]
    picture(s, figs[0], 0.6, 1.7, 8.2, 5.25)
    picture(s, figs[1], 9.0, 1.7, 3.75, 3.4)
    if q:
        text(s, 9.0, 5.25, 3.75, 1.6, f"Per-image score (TP − FP at conf ≥ 0.25) improves on {q['images_better']} and worsens on {q['images_worse']} of {q['images_compared']} test images (1–6 objects).", size=11, color=GREY)
    notes(s, "Images are chosen automatically by the largest per-image score change between baseline seed 0 and LN+GN seed 0.")


def slide_threats(prs, f: Facts):
    s = new_slide(prs, "Validity", "Threats to validity, and what was done about each")
    rows = [
        ["Threat", "What was done", "What remains"],
        ["Run-to-run noise", "3 seeds per arm; seed-paired image bootstrap (400 replicates) for every Δ", "3 seeds is a small sample of training noise"],
        ["Selecting on the test set", "Checkpoint picked on VOC07 val; test scored once per run", "—"],
        ["Confounded changes", "One factor per arm; controls for spatial layout and for the gate's init", "Adapter width (64) and α not varied"],
        ["'Not significant' ≠ 'no effect'", "Equivalence test (90% CI inside ±0.5 points); Holm over the 4 normalization comparisons", "The ±0.5 margin is a judgment call"],
        ["Implementation bugs", "17 unit tests (zero-init equality, channel offsets, resume, error types); bootstrap AP equals Ultralytics exactly", "—"],
        ["Gain bought with compute", "YOLOv8s control; DINOv2 resolution sweep; FLOPs and batched latency", "Latency on one GPU type"],
        ["One dataset", "10%-label regime; VisDrone aerial domain", "Two datasets, one ViT size"],
        ["Short schedule", "Same 15 epochs for every arm; dev curves inspected", "Longer training may narrow the gap"],
    ]
    table(s, 0.6, 1.8, 12.15, rows, [2.75, 5.7, 3.7], size=11, row_h=0.52)
    notes(s, "Quality of analysis is mostly about ruling out the boring explanations. Walk the left column and say which slide rules each one out.")


def slide_conclusion(prs, f: Facts):
    s = new_slide(prs, "Conclusions", "What the controlled study says")
    items = []
    if "ci90_map" in (f.boot_nn.get("lngn") or {}):
        verdict = "rejected. " if f.h1_rejected() else "not supported. "
        items.append([("H1 normalization — " + verdict, {"bold": True, "color": CORAL}), (f.equivalence_sentence() + ". It changes the injected scale 10×, not the accuracy.", {})])
    boot = f.boot.get("lngn")
    if boot:
        items.append([("H2 fusion — " + ("supported. " if excludes_zero(boot) else "not supported. "), {"bold": True, "color": TEAL}), (f.delta_sentence("lngn"), {})])
    cap = f.b("capacity", "lngn518_vs_yolov8s")
    b, base = f.bench.get(("yolov8n", "lngn", 518)), f.bench.get(("yolov8n", "baseline", None))
    cost = f" at {b['gflops_total'] / base['gflops_total']:.0f}× YOLOv8n's GFLOPs" if b and base else ""
    if cap:
        items.append([("H3 cost — ", {"bold": True, "color": TEAL}), (f"LN+GN vs YOLOv8s: {ci_short(cap)}{cost}.", {})])
    else:
        items.append([("H3 cost — open. ", {"bold": True, "color": TEAL}), (f"Fusion costs{cost or ' more'}; the YOLOv8s control is pending.", {})])
    low = f.b("lowdata", "lngn")
    if low and boot:
        vis = f.b("visdrone", "lngn")
        tail = f"; VisDrone {ci_short(vis)}" if vis else "; VisDrone pending"
        items.append([("H4 regime — ", {"bold": True, "color": TEAL}), (f"gain {signed(low['delta_map'])} with 10% labels vs {signed(boot['delta_map'])} with 100%{tail}.", {})])
    g = f.boot_nn.get("gated")
    if g and excludes_zero(g):
        items.append([("Extra — ", {"bold": True}), (f"the learnable per-channel gate is the only adapter change beyond noise ({signed(g['delta_map'], 2)} vs no norm).", {})])
    bullets(s, 0.6, 1.8, 7.6, 4.9, items, size=12.5, gap=10)
    label_box(
        s, 8.5, 1.8, 4.25, 4.95, "Limitations",
        [
            "• 15-epoch fine-tuning; longer schedules may narrow gaps",
            "• One small ViT (S/14) and one CNN size for fusion",
            "• VOC labels drop 'difficult' objects (Ultralytics convention)",
            "• Latency measured on one laptop GPU",
            "• Bootstrap covers test-image sampling; seeds cover training noise — neither covers other datasets",
        ],
        fill=SAND, title_color=CORAL, body_size=12,
    )
    notes(s, "Close with the one-sentence answer to the research question, then the strongest limitation.")


def slide_backup(prs, f: Facts):
    s = new_slide(prs, "Backup", "Reproducibility and anticipated questions")
    rows = [
        ["Command", "What it does"],
        ["python -m tools.prepare_data", "VOC download + fixed splits"],
        ["python -m pytest tests", "17 tests: offsets, zero-init equality, pickling, resume, gradients, error types"],
        ["python -m tools.run_queue", "every run, resumable (laptop)"],
        ["tools.kaggle_worker", "same runs on Kaggle GPUs, one per GPU"],
        ["python -m tools.analyze", "tables, figures, bootstrap CIs, error analysis"],
        ["python -m tools.make_slides", "this deck"],
    ]
    table(s, 0.6, 1.8, 6.3, rows, [2.7, 3.6], size=11, row_h=0.42)
    text(s, 0.6, 4.95, 6.3, 1.2, "Each run keeps its dev curve, diagnostics.jsonl, best.pt and per-image test matches, so every number in this deck can be regenerated.", size=12, color=GREY)
    label_box(
        s, 7.1, 1.8, 5.65, 3.6, "Likely questions",
        [
            "• Why LN+GN, not BatchNorm? → 'Why normalization could matter'",
            "• Bigger than seed noise? → 'Main result' CIs + equivalence test",
            "• Worth the compute? → 'Cost', YOLOv8s control",
            "• Did copied filters read the right channels? → 'Method', unit test",
            "• What does DINOv2 actually fix? → 'Error analysis', 'Controls'",
            "• Beyond VOC? → 'Generalization'",
        ],
        body_size=12,
    )
    notes(s, _qa_notes(f))


def _qa_notes(f: Facts) -> str:
    t = f.tokens
    ratio = ""
    if t:
        r = t["init_ctx_over_p5_rms"]
        ratio = f" At initialization the GN arms inject {r['lngn']:.3f}× the P5 scale versus {r['nonorm']:.3f}× without norm and {r['ln']:.3f}× with LN only."
    return f"""
Q: Why LayerNorm + GroupNorm, why not BatchNorm?
A: The ViT tokens and the CNN's P5 features have different scales; normalization decides how strongly DINOv2 speaks.{ratio}
GroupNorm does not depend on batch size (batch 16, and batch 1 at deployment) and adds no running statistics. ViT-S/14 shows no
high-norm outlier tokens, so the 'registers' argument is not the reason here — the scale gap is.

Q: Is the gain bigger than the noise?
A: {f.delta_sentence('lngn')}.

Q: Is it worth the latency?
A: See the YOLOv8s control and the DINO-resolution sweep on the cost slide.

Q: How do you know the copied pretrained filters read the right channels?
A: Unit test: with DINO columns zeroed the fused model's output equals the baseline's exactly (max |diff| = 0) for every arm,
and a second test silences the DINO branch after randomizing its columns.

Q: How is this different from the IUI project?
A: IUI produced one confounded, single-seed run. Here: isolated arms, fixed weight transfer, dev/test separation, 3 seeds,
bootstrap CIs, mechanism logging, cost and generalization experiments.
"""


def main() -> None:
    summary_path = RESULTS_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    f = Facts(summary)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    builders = (
        slide_title,
        slide_problem,
        slide_question,
        slide_related,
        slide_method,
        slide_protocol,
        slide_experiments,
        slide_statistics,
        slide_main,
        slide_mechanism,
        slide_controls,
        slide_curves,
        slide_where,
        slide_errors,
        slide_cost,
        slide_regimes,
        slide_qualitative,
        slide_threats,
        slide_conclusion,
        slide_backup,
    )
    global TOTAL
    TOTAL = len(builders)
    for build in builders:
        build(prs, f)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(OUT)
        print(f"saved {OUT}")
    except PermissionError:
        alt = OUT.with_name(f"{OUT.stem}_{time.strftime('%H%M%S')}.pptx")
        prs.save(alt)
        print(f"{OUT.name} is open in PowerPoint; saved {alt}")


if __name__ == "__main__":
    main()
