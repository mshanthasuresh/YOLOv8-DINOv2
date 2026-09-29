from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms as T


class ConvDummy(nn.Module):
    """Pass the original image through the YOLO graph unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DINOv2(nn.Module):
    """Frozen DINOv2 with spatial patch-token features and a trainable
    projection plus gated fusion.

    Improvements over the first version:
    1. Uses DINOv2 patch tokens (spatially varying) instead of the single
       global class token.  Patch tokens carry location-specific information
       that detection needs.
    2. Projects 384 DINO channels to 64 output channels (not 3), giving the
       downstream detector a richer context representation.
    3. Adds a learnable sigmoid gate so the network can control how much
       DINO context to inject at each channel and position.
    """

    def __init__(self, output_channels=64):
        super().__init__()
        with warnings.catch_warnings(), open(
            os.devnull, "w", encoding="utf-8"
        ) as sink:
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(
                sink
            ):
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14"
                )
        self.model.eval()
        self.model.requires_grad_(False)

        # Trainable 1x1 projection: 384 DINO channels -> output_channels
        self.projector = nn.Conv2d(384, output_channels, kernel_size=1)
        nn.init.normal_(self.projector.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projector.bias)

        # Learnable gate: starts near zero so the pretrained YOLO detector
        # is initially unaffected; the gate opens during training as the
        # projector learns useful context.
        self.gate = nn.Parameter(torch.zeros(1, output_channels, 1, 1))

        self.transform = T.Compose([
            T.Resize((518, 518), antialias=True),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, input_tensor):
        with torch.no_grad():
            processed = self.transform(input_tensor)
            # Use get_intermediate_layers to obtain spatial patch tokens
            # instead of the single global class token.
            patch_tokens = self.model.get_intermediate_layers(
                processed, n=1, reshape=True
            )[0]  # (B, 384, H_patch, W_patch)

        height = max(1, input_tensor.shape[2] // 32)
        width = max(1, input_tensor.shape[3] // 32)

        projected = self.projector(patch_tokens)
        # Gated: gate * projected, scaled by 0.1 at init for stability
        gated = torch.sigmoid(self.gate) * projected * 0.1
        return torch.nn.functional.interpolate(
            gated,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )


def register_modules() -> None:
    """Make custom names visible to Ultralytics' YAML parser."""
    hub_repositories = sorted(
        Path(torch.hub.get_dir()).glob("facebookresearch_dinov2_*")
    )
    for repository in hub_repositories:
        if (repository / "dinov2").is_dir():
            repository_path = str(repository)
            if repository_path not in sys.path:
                sys.path.insert(0, repository_path)
            break

    from ultralytics.nn import tasks

    tasks.ConvDummy = ConvDummy
    tasks.DINOv2 = DINOv2
