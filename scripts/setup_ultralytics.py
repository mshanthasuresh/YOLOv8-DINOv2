from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party" / "ultralytics"

if not TARGET.exists():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/ultralytics.git", str(TARGET)],
        check=True,
    )

subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(TARGET)], check=True)
print(f"Ultralytics editable install ready: {TARGET}")
