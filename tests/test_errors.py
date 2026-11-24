import numpy as np
import pytest

from nafuse.errors import decompose


def test_each_error_type_is_found_and_fixed(tmp_path):
    gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], np.float32)
    gt_cls = np.array([0, 1, 0])
    preds = [  # (class, box, conf) in NMS order
        (0, [0, 0, 10, 10], 0.9),  # true positive for GT 0
        (0, [0, 0, 10, 10], 0.8),  # duplicate of GT 0
        (0, [20, 20, 30, 30], 0.7),  # right box, wrong class (GT 1)
        (1, [20, 20, 26, 26], 0.6),  # right class, IoU 0.36 with GT 1
        (1, [100, 100, 110, 110], 0.5),  # background
        (1, [0, 0, 6, 6], 0.4),  # wrong class and poor box (GT 0)
    ]  # GT 2 is never detected
    path = tmp_path / "records.npz"
    np.savez(
        path,
        im=np.array(["a.jpg"]),
        conf=np.array([p[2] for p in preds], np.float32),
        conf_offsets=np.array([0, len(preds)]),
        pred_cls=np.array([p[0] for p in preds], np.int16),
        pred_boxes=np.array([p[1] for p in preds], np.float32),
        target_cls=gt_cls.astype(np.int16),
        target_cls_offsets=np.array([0, 3]),
        target_boxes=gt_boxes,
    )
    r = decompose(path)
    counts = r["counts_at_conf"]
    assert {k: counts[k] for k in ("tp", "dupe", "cls", "loc", "bkg", "both", "missed")} == {
        "tp": 1, "dupe": 1, "cls": 1, "loc": 1, "bkg": 1, "both": 1, "missed": 2,
    }
    assert r["ap50"] == pytest.approx(0.25, abs=0.01)  # class 0: AP 0.5 (1 of 2 GT found first), class 1: AP 0
    dap = r["dap50"]
    assert dap["cls"] == pytest.approx(0.5, abs=0.01)  # relabelled detection becomes class 1's TP
    assert dap["loc"] == pytest.approx(0.5, abs=0.01)  # corrected box becomes class 1's TP
    assert dap["miss"] == pytest.approx(0.25, abs=0.01)  # dropping the missed GT doubles class 0 recall
    assert dap["dupe"] == pytest.approx(0.0, abs=1e-9) and dap["bkg"] == pytest.approx(0.0, abs=1e-9)
