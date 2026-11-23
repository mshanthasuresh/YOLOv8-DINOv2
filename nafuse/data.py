"""Dataset download and split preparation.

VOC protocol (fixed for every run):
  train = VOC2012 train+val + VOC2007 train      (checkpoint training)
  dev   = VOC2007 val                            (checkpoint selection only)
  test  = VOC2007 test                           (touched once per finished run)
  train10 = seeded 10% subset of train           (low-data experiment, hard-linked into its own folder)
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

from ultralytics.utils import YAML

from . import DATA_ROOT

VOC_DIR = DATA_ROOT / "VOC"
VISDRONE_DIR = DATA_ROOT / "VisDrone"
VOC_TRAIN_DIRS = ("images/train2012", "images/val2012", "images/train2007")
VOC_DEV_DIR = "images/val2007"
VOC_TEST_DIR = "images/test2007"


def _ultralytics_dataset_cfg(name: str) -> dict:
    import ultralytics

    return YAML.load(Path(ultralytics.__file__).parent / "cfg" / "datasets" / name)


def _download_with_builtin_script(name: str, root: Path) -> None:
    """Run Ultralytics' own download/convert script for a built-in dataset into ``root``."""
    from ultralytics.data.utils import check_det_dataset

    cfg = _ultralytics_dataset_cfg(name)
    cfg["path"] = str(root)
    tmp_yaml = DATA_ROOT / f"_download_{Path(name).stem}.yaml"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    YAML.save(tmp_yaml, cfg)
    check_det_dataset(str(tmp_yaml), autodownload=True)
    tmp_yaml.unlink(missing_ok=True)


def download_voc() -> None:
    if (VOC_DIR / VOC_TEST_DIR).is_dir() and any((VOC_DIR / VOC_TEST_DIR).iterdir()):
        return
    _download_with_builtin_script("VOC.yaml", VOC_DIR)


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_voc_subset(fraction: float, seed: int = 0, name: str | None = None) -> str:
    """Hard-link a seeded fraction of the VOC train split into images/<name> and labels/<name>."""
    name = name or f"train{round(fraction * 100)}"
    images = [p for d in VOC_TRAIN_DIRS for p in _images_in(VOC_DIR / d)]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(images, max(1, round(len(images) * fraction))))
    img_out, lbl_out = VOC_DIR / "images" / name, VOC_DIR / "labels" / name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img in chosen:
        label = VOC_DIR / "labels" / img.parent.name / f"{img.stem}.txt"
        _link_or_copy(img, img_out / img.name)
        if label.exists():
            _link_or_copy(label, lbl_out / label.name)
    return f"images/{name}"


def write_voc_yamls(low_fraction: float = 0.1, seed: int = 0) -> dict[str, Path]:
    names = _ultralytics_dataset_cfg("VOC.yaml")["names"]
    subset_dir = make_voc_subset(low_fraction, seed=seed)
    common = {"path": str(VOC_DIR), "val": VOC_DEV_DIR, "test": VOC_TEST_DIR, "names": names}
    out = {
        "voc": DATA_ROOT / "voc.yaml",
        "voc10": DATA_ROOT / "voc10.yaml",
    }
    YAML.save(out["voc"], {**common, "train": list(VOC_TRAIN_DIRS)})
    YAML.save(out["voc10"], {**common, "train": subset_dir})
    return out


def download_visdrone() -> None:
    if (VISDRONE_DIR / "images" / "test").is_dir():
        return
    _download_with_builtin_script("VisDrone.yaml", VISDRONE_DIR)


def write_visdrone_yaml() -> Path:
    cfg = _ultralytics_dataset_cfg("VisDrone.yaml")
    out = DATA_ROOT / "visdrone.yaml"
    YAML.save(
        out,
        {
            "path": str(VISDRONE_DIR),
            "train": cfg["train"],
            "val": cfg["val"],
            "test": cfg["test"],
            "names": cfg["names"],
        },
    )
    return out


def dataset_yaml(name: str) -> Path:
    path = DATA_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m tools.prepare_data` first")
    return path
