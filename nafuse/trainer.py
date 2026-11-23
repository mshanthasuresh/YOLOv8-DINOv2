"""Ultralytics trainer/validator subclasses: fused model construction, fusion diagnostics, per-image records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import RANK, ops
from ultralytics.utils.torch_utils import unwrap_model

from .adapter import FusionAdapter, _rms
from .model import FusedDetectionModel, attach_fusion


def parameter_stats(model: FusedDetectionModel) -> dict[str, float]:
    fusion = model.fusion
    stats = {"proj_w_rms": _rms(fusion.proj.weight)}
    if fusion.token_norm is not None:
        stats["ln_w_mean"] = float(fusion.token_norm.weight.detach().float().mean())
    if fusion.out_norm is not None:
        stats["gn_w_mean"] = float(fusion.out_norm.weight.detach().float().mean())
    if fusion.gate is not None:
        stats["gate_mean"] = float(torch.sigmoid(fusion.gate.detach().float()).mean())
    for conv in model.split_convs():
        w_dino, w_yolo = conv.split_weights()
        stats[f"{conv.tag}/w_rms_dino"] = _rms(w_dino)
        stats[f"{conv.tag}/w_rms_yolo"] = _rms(w_yolo)
    return stats


class FusionTrainer(DetectionTrainer):
    """DetectionTrainer that builds :class:`FusedDetectionModel` (``fusion=None`` gives the matched baseline)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, fusion: dict | None = None, diag_every=50):
        self.fusion_cfg = fusion
        self.diag_every = diag_every
        self._diag_iter = -1
        super().__init__(cfg, overrides, _callbacks)
        if fusion:
            self.add_callback("on_train_batch_start", _diagnostics_start)
            self.add_callback("on_train_batch_end", _diagnostics_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, FusedDetectionModel):
            restored_fusion = getattr(weights, "fusion", None)
            if self.fusion_cfg:
                if restored_fusion is None:
                    raise ValueError("resume checkpoint has no fusion adapter")
                restored = restored_fusion.config()
                for key in ("arm", "resolution", "out_channels"):
                    expected = self.fusion_cfg.get(key, 64 if key == "out_channels" else None)
                    if restored.get(key) != expected:
                        raise ValueError(f"resume checkpoint fusion mismatch for {key}: {restored.get(key)}")
                restored_fusion.recorder.active = False
                for conv in weights.split_convs():
                    conv.recorder = restored_fusion.recorder
            elif restored_fusion is not None:
                raise ValueError("baseline run cannot resume from a fused checkpoint")
            return self.set_model_names_for_load(weights)

        model = FusedDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        if self.fusion_cfg:
            attach_fusion(model, FusionAdapter(**self.fusion_cfg))
        return model


def _diagnostics_start(trainer: FusionTrainer) -> None:
    trainer._diag_iter += 1
    if trainer._diag_iter % trainer.diag_every == 0:
        recorder = unwrap_model(trainer.model).fusion.recorder
        recorder.values = {}
        recorder.active = True


def _diagnostics_end(trainer: FusionTrainer) -> None:
    model = unwrap_model(trainer.model)
    recorder = model.fusion.recorder
    if not recorder.active:
        return
    recorder.active = False
    row = {"epoch": trainer.epoch, "iter": trainer._diag_iter, **recorder.values, **parameter_stats(model)}
    with open(Path(trainer.save_dir) / "diagnostics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


class RecordingValidator(DetectionValidator):
    """DetectionValidator that also keeps per-image matches for paired bootstrap and size analysis."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            tp = self._process_batch(pred, pbatch)["tp"]
            gt = ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"])
            pb = ops.scale_boxes(pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"])
            self.records.append(
                {
                    "im": Path(pbatch["im_file"]).name,
                    "tp": tp.astype(bool),
                    "conf": pred["conf"].float().cpu().numpy(),
                    "pred_cls": pred["cls"].cpu().numpy().astype(np.int16),
                    "pred_area": ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])).float().cpu().numpy(),
                    "target_cls": pbatch["cls"].cpu().numpy().astype(np.int16),
                    "target_area": ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).float().cpu().numpy(),
                    "pred_boxes": pb.float().cpu().numpy(),
                    "target_boxes": gt.float().cpu().numpy(),
                }
            )

    def save_records(self, path: Path) -> None:
        def ragged(key):
            arrays = [r[key] for r in self.records]
            return np.concatenate(arrays) if arrays else np.zeros(0), np.cumsum([0] + [len(a) for a in arrays])

        out = {"im": np.array([r["im"] for r in self.records])}
        for key in ("tp", "conf", "pred_cls", "pred_area", "pred_boxes", "target_cls", "target_area", "target_boxes"):
            out[key], out[f"{key}_offsets"] = ragged(key)
        np.savez_compressed(path, **out)
