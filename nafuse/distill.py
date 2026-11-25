"""Feature distillation: plain YOLOv8n learns to predict DINOv2's patch features at P5 during training only.

At inference the head is never run, so the deployed model is exactly YOLOv8n (same FLOPs and latency).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dino import FrozenDINO


class DistillHead(nn.Module):
    """1x1 conv from P5 to DINOv2's token space; loss = 1 - cosine similarity to the frozen teacher's tokens."""

    def __init__(self, in_channels: int, resolution: int = 518, weight: float = 1.0, backbone: str = "dinov2_vits14"):
        super().__init__()
        self.dino = FrozenDINO(backbone, resolution)
        self.weight = weight
        self.proj = nn.Conv2d(in_channels, self.dino.embed_dim, kernel_size=1)

    def config(self) -> dict:
        return {"resolution": self.dino.resolution, "weight": self.weight, "backbone": self.dino.name}

    def forward(self, p5: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(self.dino.tokens(images), size=p5.shape[-2:], mode="bilinear", align_corners=False)
        pred = self.proj(p5).float()
        return (1 - F.cosine_similarity(pred, target, dim=1)).mean()
