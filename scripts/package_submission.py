from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Create an FTP-ready source archive.")
parser.add_argument("--output", default=str(ROOT.parent / "submission_yolov8_dinov2.zip"))
args = parser.parse_args()

archive = Path(args.output).with_suffix("")
if archive.exists():
    shutil.rmtree(archive)
shutil.copytree(
    ROOT,
    archive,
    ignore=shutil.ignore_patterns(
        ".venv", ".venv-wsl", "__pycache__", "runs", "third_party", "data", "weights", "*.pt"
    ),
)
artifact_sources = {
    ROOT / "REPORT_TEMPLATE.md": archive / "REPORT_TEMPLATE.md",
    ROOT / "runs" / "baseline_pretrained_metrics" / "metrics.json": archive / "artifacts" / "baseline_pretrained_metrics.json",
    ROOT / "runs" / "trained_metrics" / "metrics.json": archive / "artifacts" / "trained_metrics.json",
    ROOT / "runs" / "dino_inference" / "dino_prediction.jpg": archive / "artifacts" / "dino_prediction.jpg",
    ROOT / "runs" / "pybullet_adas" / "adas_replay.mp4": archive / "artifacts" / "adas_replay.mp4",
    ROOT / "runs" / "pybullet_adas" / "adas_replay.csv": archive / "artifacts" / "adas_replay.csv",
}
for source, destination in artifact_sources.items():
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
shutil.make_archive(str(archive), "zip", root_dir=archive)
shutil.rmtree(archive)
print(f"FTP-ready archive: {archive}.zip")
