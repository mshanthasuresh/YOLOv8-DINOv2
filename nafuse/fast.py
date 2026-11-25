"""Inference-only speed-up that leaves the network unchanged: CUDA-graph replay.

At batch 1 a laptop GPU spends most of the fused forward pass waiting for the CPU to launch hundreds of small
kernels. Capturing the pass once and replaying it as a single CUDA graph removes that overhead while running
exactly the same kernels on the same data.
"""

from __future__ import annotations

import torch


def first_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class GraphedModel:
    """Captures ``model(x)`` for one fixed input shape and replays it. Outputs are reused buffers: copy them
    before the next call if they must be kept."""

    def __init__(self, model: torch.nn.Module, example: torch.Tensor, warmup: int = 3):
        self.static_in = example.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(warmup):  # also builds lazy state such as the detection head's anchor grid
                model(self.static_in)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_out = model(self.static_in)

    def __call__(self, x: torch.Tensor):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out
