"""YOLOv8 detection model with an optional DINOv2 context branch fused after SPPF.

Fusion layout: the P5 (SPPF) output becomes ``cat([p5, ctx])`` - YOLO channels first, DINO channels appended.
Every concatenation that carries P5 therefore grows by ``extra`` channels right after the P5 block, and the
first convolution after it is widened at exactly that offset. Pretrained filters keep reading their original
channels; the new DINO columns are zero-initialized, so the untrained fused model reproduces the baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Concat
from ultralytics.nn.tasks import DetectionModel

from .adapter import FusionAdapter, SplitConv2d
from .distill import DistillHead


class FusedDetectionModel(DetectionModel):
    def _predict_once(self, x, profile=False, embed=None):
        fusion = getattr(self, "fusion", None)
        distill = getattr(self, "distill", None)
        if fusion is None and distill is None:
            return super()._predict_once(x, profile, embed)
        image = x
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if m.i == self.fuse_index:
                if distill is not None and self.training:
                    self._p5 = x
                if fusion is not None:
                    x = self._fuse(image, x)
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _fuse(self, image: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        ctx = self.fusion(image, p5.shape[-2:]).to(p5.dtype)
        fused = torch.cat([p5, ctx], 1)
        recorder = self.fusion.recorder
        if recorder.active:
            recorder.record_features(p5, ctx)
            if fused.requires_grad:
                c = p5.shape[1]
                fused.register_hook(lambda g: recorder.record_grads(g[:, :c], g[:, c:]))
        return fused

    def split_convs(self) -> list[SplitConv2d]:
        return [m for m in self.modules() if isinstance(m, SplitConv2d)]

    def loss(self, batch, preds=None):
        distill = getattr(self, "distill", None)
        if distill is None:
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch["img"])
        loss, items = super().loss(batch, preds)
        p5, self._p5 = getattr(self, "_p5", None), None
        d = distill(p5, batch["img"]) if self.training and p5 is not None else loss.new_zeros(())
        weighted = (distill.weight * batch["img"].shape[0] * d).reshape(1)  # same batch scaling as the box/cls/dfl terms
        return torch.cat([loss, weighted]), {**items, "dino_loss": d.detach()}


@torch.no_grad()
def layer_out_channels(model: DetectionModel) -> dict[int, int]:
    """Output channel count of every layer except the head, from a tiny dry run."""
    channels: dict[int, int] = {}
    hooks = [
        m.register_forward_hook(lambda mod, inp, out: channels.__setitem__(mod.i, out.shape[1]))
        for m in model.model[:-1]
    ]
    was_training = model.training
    try:
        model.eval()
        p = next(model.parameters())
        model._predict_once(torch.zeros(1, 3, 64, 64, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return channels


def attach_fusion(model: FusedDetectionModel, adapter: FusionAdapter, fuse_index: int | None = None):
    """Insert ``adapter`` after layer ``fuse_index`` (default: last backbone layer, i.e. SPPF) and widen consumers."""
    if getattr(model, "fusion", None) is not None:
        raise RuntimeError("fusion already attached")
    fuse_index = len(model.yaml["backbone"]) - 1 if fuse_index is None else fuse_index
    channels = layer_out_channels(model)
    c_p5, extra = channels[fuse_index], adapter.out_channels
    carriers, layout = {fuse_index}, []
    for m in model.model[:-1]:
        inputs = [m.f] if isinstance(m.f, int) else list(m.f)
        inputs = [m.i - 1 if j == -1 else j for j in inputs]
        if not any(j in carriers for j in inputs):
            continue
        if isinstance(m, nn.Upsample) and len(inputs) == 1:
            carriers.add(m.i)
        elif isinstance(m, Concat):
            offset, pos = 0, None
            for j in inputs:
                if j in carriers:
                    pos = offset + c_p5
                offset += channels[j]
            consumer = model.model[m.i + 1]
            if consumer.f != -1 or not hasattr(consumer, "cv1"):
                raise NotImplementedError(f"layer {consumer.i} ({consumer.type}) cannot absorb fused channels")
            conv = consumer.cv1.conv
            if conv.in_channels != offset:
                raise RuntimeError(f"layer {consumer.i}: expected {offset} input channels, found {conv.in_channels}")
            consumer.cv1.conv = SplitConv2d.widen(conv, pos, extra, adapter.recorder, tag=f"L{consumer.i}")
            layout.append({"concat": m.i, "consumer": consumer.i, "pos": pos, "extra": extra, "in_before": offset})
        else:
            raise NotImplementedError(f"layer {m.i} ({m.type}) consumes the fused P5 tensor")
    if not layout:
        raise RuntimeError("no concatenation consumes the fusion point")
    model.fusion = adapter
    model.fuse_index = fuse_index
    model.fusion_layout = layout
    model.yaml = {**model.yaml, "fusion": adapter.config()}
    return model


def attach_distill(model: FusedDetectionModel, head_cfg: dict, fuse_index: int | None = None):
    """Training-only DINOv2 feature head on the SPPF output; the inference graph stays plain YOLO."""
    fuse_index = len(model.yaml["backbone"]) - 1 if fuse_index is None else fuse_index
    model.distill = DistillHead(layer_out_channels(model)[fuse_index], **head_cfg)
    model.fuse_index = fuse_index
    model.yaml = {**model.yaml, "distill": model.distill.config()}
    return model


def build_model(
    cfg: str | dict = "yolov8n.yaml",
    nc: int = 20,
    fusion: dict | None = None,
    weights=None,
    names: dict | None = None,
    verbose: bool = False,
) -> FusedDetectionModel:
    """Same construction order as training: build, load pretrained weights, then attach fusion."""
    model = FusedDetectionModel(cfg, nc=nc, ch=3, verbose=verbose)
    if names:
        model.names = names
    if weights is not None:
        model.load(weights, verbose=verbose)
    if fusion:
        attach_fusion(model, FusionAdapter(**fusion))
    return model
