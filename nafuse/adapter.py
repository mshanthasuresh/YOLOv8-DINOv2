"""DINOv2 -> YOLO fusion adapter variants and the training-time diagnostics recorder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .dino import FrozenDINO

# Each arm changes exactly one design factor relative to the supplied design ("lngn").
ARMS: dict[str, dict] = {
    "nonorm": {"token_norm": False, "out_norm": False, "gated": False},
    "ln": {"token_norm": True, "out_norm": False, "gated": False},
    "gn": {"token_norm": False, "out_norm": True, "gated": False},
    "lngn": {"token_norm": True, "out_norm": True, "gated": False},
    # Historical IUI variant, reproduced as-is: no normalization, per-channel sigmoid gate, std-0.02 projector.
    "gated": {"token_norm": False, "out_norm": False, "gated": True},
}
ARM_LABELS = {
    "baseline": "YOLOv8n baseline",
    "nonorm": "No adapter norm",
    "ln": "Token LayerNorm only",
    "gn": "Projected GroupNorm only",
    "lngn": "LayerNorm + GroupNorm (supplied)",
    "gated": "Gated fusion (historical)",
}


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


class DiagnosticsRecorder:
    """Collects fusion statistics on the iterations where ``active`` is set; inert otherwise."""

    def __init__(self):
        self.active = False
        self.values: dict[str, float] = {}

    def __getstate__(self):
        return {"active": False, "values": {}}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def record_features(self, yolo: torch.Tensor, ctx: torch.Tensor) -> None:
        y, c = _rms(yolo), _rms(ctx)
        self.values.update(feat_rms_yolo=y, feat_rms_dino=c, feat_ratio=c / max(y, 1e-12))

    def record_grads(self, g_yolo: torch.Tensor, g_ctx: torch.Tensor) -> None:
        y, c = _rms(g_yolo), _rms(g_ctx)
        if not (math.isfinite(y) and math.isfinite(c)):
            return  # AMP overflow step; GradScaler skips it too
        # GradScaler multiplies both parts by the same factor, so only the ratio is meaningful.
        self.values.update(grad_ratio=c / max(y, 1e-30))


class FusionAdapter(nn.Module):
    """tokens -> [LayerNorm] -> 1x1 projection -> [GroupNorm] -> scale (fixed or gated) -> resize to P5 grid."""

    def __init__(
        self,
        arm: str = "lngn",
        out_channels: int = 64,
        resolution: int = 518,
        scale: float | None = None,
        proj_std: float | None = None,
        backbone: str = "dinov2_vits14",
        gn_groups: int = 1,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown fusion arm {arm!r}; choose from {sorted(ARMS)}")
        flags = ARMS[arm]
        self.arm = arm
        self.dino = FrozenDINO(backbone, resolution)
        dim = self.dino.embed_dim
        self.out_channels = out_channels
        self.gated = flags["gated"]
        self.scale = scale if scale is not None else (0.1 if self.gated else 0.05)
        self.proj_std = proj_std if proj_std is not None else (0.02 if self.gated else 0.005)
        self.gn_groups = gn_groups
        self.token_norm = nn.LayerNorm(dim) if flags["token_norm"] else None
        self.proj = nn.Conv2d(dim, out_channels, kernel_size=1)
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.proj_std)
        nn.init.zeros_(self.proj.bias)
        self.out_norm = nn.GroupNorm(gn_groups, out_channels) if flags["out_norm"] else None
        self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1)) if self.gated else None
        self.recorder = DiagnosticsRecorder()

    def config(self) -> dict:
        return {
            "arm": self.arm,
            "out_channels": self.out_channels,
            "resolution": self.dino.resolution,
            "scale": self.scale,
            "proj_std": self.proj_std,
            "backbone": self.dino.name,
            "gn_groups": self.gn_groups,
        }

    def effective_scale(self) -> torch.Tensor | float:
        return torch.sigmoid(self.gate) * self.scale if self.gated else self.scale

    def forward(self, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        tokens = self.dino.tokens(images).to(self.proj.weight.dtype)
        if self.token_norm is not None:
            tokens = self.token_norm(tokens.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ctx = self.proj(tokens)
        if self.out_norm is not None:
            ctx = self.out_norm(ctx)
        ctx = ctx * self.effective_scale()
        return F.interpolate(ctx, size=tuple(size), mode="bilinear", align_corners=False)


class SplitConv2d(nn.Conv2d):
    """A neck convolution whose input gained DINO channels at ``dino_slice``; can report the DINO share."""

    @classmethod
    def widen(cls, conv: nn.Conv2d, pos: int, extra: int, recorder: DiagnosticsRecorder, tag: str) -> SplitConv2d:
        if conv.groups != 1:
            raise NotImplementedError("grouped convolutions are not supported")
        new = cls(
            conv.in_channels + extra,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            new.weight.zero_()  # DINO columns start at zero: the fused model equals the baseline at step 0
            new.weight[:, :pos] = conv.weight[:, :pos]
            new.weight[:, pos + extra :] = conv.weight[:, pos:]
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        new.dino_slice = (pos, pos + extra)
        new.recorder = recorder
        new.tag = tag
        return new

    def split_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.dino_slice
        return self.weight[:, a:b], torch.cat([self.weight[:, :a], self.weight[:, b:]], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        recorder = getattr(self, "recorder", None)
        if recorder is not None and recorder.active:
            a, b = self.dino_slice
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                w = self.weight.float()
                x32 = x.float()
                y_dino = F.conv2d(x32[:, a:b], w[:, a:b], None, self.stride, self.padding, self.dilation)
                y_yolo = F.conv2d(
                    torch.cat([x32[:, :a], x32[:, b:]], 1),
                    torch.cat([w[:, :a], w[:, b:]], 1),
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                )
                d, o = _rms(y_dino), _rms(y_yolo)
                recorder.values[f"{self.tag}/dino_share"] = d / max(d + o, 1e-12)
        return y
