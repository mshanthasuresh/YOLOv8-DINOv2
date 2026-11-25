"""Frozen DINOv2 backbone shared by every fused model.

The backbone is deliberately *not* registered as a submodule: it never enters the state_dict, the optimizer,
Ultralytics' unfreeze pass, the EMA copy or saved checkpoints. It is loaded lazily (from the local torch.hub
cache when available) on whichever device the images arrive on.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

PATCH = 14
EMBED_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_BACKBONES: dict[tuple[str, str], torch.nn.Module] = {}
_NORM_STATS: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def load_backbone(name: str = "dinov2_vits14", device: torch.device | str = "cpu") -> torch.nn.Module:
    key = (name, str(device))
    if key not in _BACKBONES:
        local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            if local_repo.is_dir():
                model = torch.hub.load(str(local_repo), name, source="local")
            else:
                model = torch.hub.load("facebookresearch/dinov2", name)
        _BACKBONES[key] = model.eval().requires_grad_(False).to(device)
    return _BACKBONES[key]


def dino_input_size(height: int, width: int, resolution: int) -> tuple[int, int]:
    """Long side -> ``resolution``; both sides snapped to the patch grid so aspect ratio is preserved."""
    scale = resolution / max(height, width)
    return (
        max(PATCH, round(height * scale / PATCH) * PATCH),
        max(PATCH, round(width * scale / PATCH) * PATCH),
    )


class FrozenDINO:
    """Handle to a shared frozen backbone; deep-copies to itself and pickles without weights.

    ``depth`` keeps only the first k transformer blocks (``None`` = all of them).
    """

    depth: int | None = None  # class default for handles pickled before the option existed

    def __init__(self, name: str = "dinov2_vits14", resolution: int = 518, depth: int | None = None):
        self.name = name
        self.resolution = resolution
        self.depth = depth

    @property
    def embed_dim(self) -> int:
        return EMBED_DIM[self.name]

    def __deepcopy__(self, memo):
        return self

    def __getstate__(self):
        return {"name": self.name, "resolution": self.resolution, "depth": self.depth}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """RGB images in [0, 1] -> ImageNet-normalized, aspect-preserving, patch-aligned input."""
        size = dino_input_size(images.shape[-2], images.shape[-1], self.resolution)
        x = F.interpolate(images.float(), size=size, mode="bilinear", align_corners=False, antialias=True)
        key = (str(x.device), x.dtype)
        if key not in _NORM_STATS:  # cached so that a CUDA-graph capture never needs a host-to-device copy
            _NORM_STATS[key] = (x.new_tensor(_MEAN).view(1, 3, 1, 1), x.new_tensor(_STD).view(1, 3, 1, 1))
        mean, std = _NORM_STATS[key]
        return (x - mean) / std

    @torch.no_grad()
    def tokens(self, images: torch.Tensor, final_norm: bool = True) -> torch.Tensor:
        """Patch tokens (B, C, h, w) from the last block; ``final_norm`` applies DINOv2's own output LayerNorm."""
        backbone = load_backbone(self.name, images.device)
        x = self.preprocess(images)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=images.is_cuda):
            if self.depth is None:
                out = backbone.get_intermediate_layers(x, n=1, reshape=True, norm=final_norm)[0]
            else:
                out = _truncated_tokens(backbone, x, self.depth, final_norm)
        return out.float()


def _truncated_tokens(backbone: torch.nn.Module, x: torch.Tensor, depth: int, final_norm: bool) -> torch.Tensor:
    """Same output layout as ``get_intermediate_layers(n=1, reshape=True)``, but blocks after ``depth`` never run."""
    b, _, h, w = x.shape
    t = backbone.prepare_tokens_with_masks(x)
    for block in backbone.blocks[:depth]:
        t = block(t)
    if final_norm:
        t = backbone.norm(t)
    t = t[:, 1 + backbone.num_register_tokens :]
    return t.reshape(b, h // backbone.patch_size, w // backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
