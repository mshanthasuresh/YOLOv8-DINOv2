from __future__ import annotations

import copy
import pickle

import pytest
import torch
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML

from nafuse import PROJECT_ROOT
from nafuse.adapter import ARMS, FusionAdapter, SplitConv2d
from nafuse.dino import FrozenDINO, dino_input_size
from nafuse.fast import GraphedModel
from nafuse.model import FusedDetectionModel, attach_distill, attach_fusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_RES = 56  # 4x4 DINO tokens keeps the tests fast; geometry logic is resolution independent


def _first_tensor(out):
    while isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return out


@pytest.fixture(scope="module")
def baseline():
    weights, _ = load_checkpoint(PROJECT_ROOT / "yolov8n.pt")
    model = FusedDetectionModel(copy.deepcopy(weights.yaml), nc=20, verbose=False)
    model.load(weights, verbose=False)
    return model.to(DEVICE).eval()


def _fused(baseline, arm="lngn"):
    model = copy.deepcopy(baseline)
    attach_fusion(model, FusionAdapter(arm, resolution=TEST_RES).to(DEVICE))
    return model.to(DEVICE).eval()


def _images(n=2, h=320, w=256):
    torch.manual_seed(0)
    return torch.rand(n, 3, h, w, device=DEVICE)


def test_layout_matches_yolov8n_neck(baseline):
    model = _fused(baseline)
    assert model.fuse_index == 9
    assert model.fusion_layout == [
        {"concat": 11, "consumer": 12, "pos": 256, "extra": 64, "in_before": 384},
        {"concat": 20, "consumer": 21, "pos": 384, "extra": 64, "in_before": 384},
    ]


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_untrained_fusion_reproduces_baseline(baseline, arm):
    model = _fused(baseline, arm)
    x = _images()
    with torch.no_grad():
        ref, out = _first_tensor(baseline(x)), _first_tensor(model(x))
    assert torch.equal(ref, out), f"{arm}: max |diff| {float((ref - out).abs().max())}"


def test_pretrained_filters_read_original_channels(baseline):
    model = _fused(baseline)
    with torch.no_grad():
        for conv in model.split_convs():
            a, b = conv.dino_slice
            conv.weight[:, a:b].normal_(0, 0.5)
    x = _images()
    with torch.no_grad():
        ref = _first_tensor(baseline(x))
        changed = _first_tensor(model(x))
        model.fusion.forward = lambda images, size: torch.zeros(
            images.shape[0], 64, *size, device=images.device, dtype=images.dtype
        )
        silenced = _first_tensor(model(x))
    assert not torch.allclose(ref, changed), "non-zero DINO columns must change the output"
    assert torch.equal(ref, silenced), "with a silent DINO branch the YOLO filters must see their original inputs"


def test_widened_weights_are_exact_copies(baseline):
    model = _fused(baseline)
    for entry in model.fusion_layout:
        old = baseline.model[entry["consumer"]].cv1.conv.weight
        new = model.model[entry["consumer"]].cv1.conv
        assert isinstance(new, SplitConv2d)
        pos, extra = entry["pos"], entry["extra"]
        assert torch.equal(new.weight[:, :pos], old[:, :pos])
        assert torch.equal(new.weight[:, pos + extra :], old[:, pos:])
        assert not new.weight[:, pos : pos + extra].any()


def test_dino_backbone_stays_out_of_state_and_checkpoints(baseline):
    model = _fused(baseline)
    keys = model.state_dict().keys()
    assert not any("blocks" in k or "patch_embed" in k for k in keys)
    adapter = sum(p.numel() for p in model.fusion.parameters())
    added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert added == adapter + 64 * (model.model[12].cv1.conv.out_channels + model.model[21].cv1.conv.out_channels)
    assert copy.deepcopy(model).fusion.dino is model.fusion.dino
    blob = pickle.dumps(model.cpu())
    assert len(blob) < 30e6, "checkpoint must not embed DINOv2 weights"
    restored = pickle.loads(blob).to(DEVICE).eval()
    model.to(DEVICE)
    x = _images()
    with torch.no_grad():
        assert torch.equal(_first_tensor(model(x)), _first_tensor(restored(x)))


def test_resume_reuses_serialized_fusion_weights(baseline):
    from ultralytics.cfg import get_cfg

    from nafuse.trainer import FusionTrainer

    model = _fused(baseline)
    with torch.no_grad():
        model.fusion.proj.weight.fill_(0.125)
        for conv in model.split_convs():
            a, b = conv.dino_slice
            conv.weight[:, a:b].fill_(0.25)

    trainer = object.__new__(FusionTrainer)
    trainer.fusion_cfg = {"arm": "lngn", "resolution": TEST_RES}
    trainer.args = get_cfg()
    trainer.data = {"names": model.names}
    restored = trainer.get_model(cfg=model.yaml, weights=model)

    assert restored is model
    assert torch.all(model.fusion.proj.weight == 0.125)
    assert all(torch.all(c.weight[:, c.dino_slice[0] : c.dino_slice[1]] == 0.25) for c in model.split_convs())
    assert all(not c.recorder.active for c in model.split_convs())


def test_adapter_learns_after_first_step(baseline):
    from ultralytics.cfg import get_cfg

    model = _fused(baseline).train()
    model.args = get_cfg()
    batch = {
        "img": _images(2, 256, 256),
        "batch_idx": torch.tensor([0.0, 1.0], device=DEVICE),
        "cls": torch.tensor([[3.0], [7.0]], device=DEVICE),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.4, 0.6, 0.2, 0.2]], device=DEVICE),
    }
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss, _ = model.loss(batch)
    loss.sum().backward()
    proj, slice_grad = model.fusion.proj.weight.grad, model.model[12].cv1.conv.weight.grad[:, 256:320]
    assert proj is not None and not proj.any(), "zero DINO columns block adapter gradients at step 0"
    assert slice_grad.abs().sum() > 0, "DINO columns must receive gradient at step 0"
    opt.step()
    opt.zero_grad()
    loss, _ = model.loss(batch)
    loss.sum().backward()
    assert model.fusion.proj.weight.grad.abs().sum() > 0, "adapter must receive gradient after one update"


def test_diagnostics_are_recorded(baseline):
    from ultralytics.cfg import get_cfg

    model = _fused(baseline).train()
    model.args = get_cfg()
    with torch.no_grad():
        for conv in model.split_convs():
            a, b = conv.dino_slice
            conv.weight[:, a:b].normal_(0, 0.01)
    batch = {
        "img": _images(2, 256, 256),
        "batch_idx": torch.tensor([0.0, 1.0], device=DEVICE),
        "cls": torch.tensor([[3.0], [7.0]], device=DEVICE),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.4, 0.6, 0.2, 0.2]], device=DEVICE),
    }
    recorder = model.fusion.recorder
    recorder.active = True
    loss, _ = model.loss(batch)
    loss.sum().backward()
    recorder.active = False
    for key in ("feat_ratio", "grad_ratio", "L12/dino_share", "L21/dino_share"):
        assert key in recorder.values and recorder.values[key] >= 0, key
    assert 0 < recorder.values["L12/dino_share"] < 1


def test_aspect_preserving_dino_resize():
    assert dino_input_size(640, 640, 518) == (518, 518)
    assert dino_input_size(480, 640, 518) == (392, 518)
    assert dino_input_size(640, 320, 224) == (224, 112)


def test_split_yaml_protocol_is_disjoint():
    voc = PROJECT_ROOT / "data" / "voc.yaml"
    if not voc.exists():
        pytest.skip("VOC not prepared yet")
    cfg = YAML.load(voc)
    train = set(cfg["train"])
    assert cfg["val"] not in train and cfg["test"] not in train and cfg["val"] != cfg["test"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_replay_is_bit_identical(baseline):
    model = _fused(baseline).half()
    x = torch.rand(1, 3, 64, 64, device="cuda", dtype=torch.half)
    with torch.inference_mode():
        eager = _first_tensor(model(x)).clone()
    assert torch.equal(_first_tensor(GraphedModel(model, x)(x)), eager)


def test_adapter_pickled_before_pool_flag_still_runs():
    adapter = FusionAdapter("lngn", resolution=TEST_RES).to(DEVICE)
    del adapter.__dict__["pool"]  # checkpoints saved before the "global" control lack this attribute
    out = adapter(torch.rand(1, 3, 64, 64, device=DEVICE), (2, 2))
    assert out.shape == (1, 64, 2, 2)


def test_truncated_dino_matches_full_and_skips_blocks():
    x = _images(1, 64, 64)
    full = FrozenDINO(resolution=TEST_RES).tokens(x)
    assert torch.allclose(FrozenDINO(resolution=TEST_RES, depth=12).tokens(x), full, atol=1e-4)
    short = FrozenDINO(resolution=TEST_RES, depth=4).tokens(x)
    assert short.shape == full.shape and not torch.allclose(short, full, atol=1e-2)
    old = pickle.loads(pickle.dumps(FrozenDINO(resolution=TEST_RES)))
    del old.__dict__["depth"]  # handles pickled before the option existed
    assert torch.equal(old.tokens(x), full)


def test_distillation_trains_head_and_backbone_but_not_inference(baseline):
    from ultralytics.cfg import get_cfg

    x = _images(1, 256, 256)
    with torch.no_grad():
        reference = _first_tensor(baseline(x))
    model = attach_distill(copy.deepcopy(baseline), {"resolution": TEST_RES}).to(DEVICE).eval()
    with torch.no_grad():
        assert torch.equal(_first_tensor(model(x)), reference), "the head must not change inference"
    model.train()
    model.args = get_cfg()
    batch = {
        "img": _images(2, 256, 256),
        "batch_idx": torch.tensor([0.0, 1.0], device=DEVICE),
        "cls": torch.tensor([[3.0], [7.0]], device=DEVICE),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.4, 0.6, 0.2, 0.2]], device=DEVICE),
    }
    loss, items = model.loss(batch)
    assert loss.shape == (4,) and items["dino_loss"] > 0
    loss.sum().backward()
    assert model.distill.proj.weight.grad.abs().sum() > 0
    assert model.model[9].cv2.conv.weight.grad.abs().sum() > 0, "distillation must reach the backbone (SPPF)"
