# YOLOv8 + DINOv2 Object Detection

This project extends YOLOv8 with a frozen DINOv2 image embedding. It is based on the computer-vision resource page supplied for the course.

## What it does

- Uses YOLOv8 for fast object detection.
- Uses DINOv2 for global image semantics.
- Injects the DINOv2 embedding into the deepest YOLO feature map.
- Supports COCO8 for a small, quick smoke test and a custom YOLO-format dataset.

## Folder layout

```text
configs/       Model and dataset configuration files
data/          Local dataset location; large files are ignored by git
scripts/       Dataset preparation, training, and prediction scripts
weights/       Local model weights; large files are ignored by git
```

## Windows setup

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\setup_ultralytics.py
```

## WSL2 setup and demonstration

The project also runs in WSL2 Ubuntu 22.04 with NVIDIA GPU passthrough. From
PowerShell, open the project in WSL and install the Linux dependencies:

```powershell
wsl -d Ubuntu-22.04
cd "/mnt/d/M.Tech/AI/4_Sem/Intelligent User Interface/Project/yolov8-dinov2-detection"
sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
nvidia-smi
```

Run the demonstration on the included COCO8 validation image:

```bash
bash scripts/demo_wsl.sh
```

The script uses the trained checkpoint when it exists, otherwise it uses the
shared `yolov8n.pt` checkpoint. It runs DINOv2 followed by YOLOv8 and writes
`runs/dino_inference/wsl_demo.jpg`. Open the result from Windows at:

```text
\\wsl$\Ubuntu-22.04\mnt\d\M.Tech\AI\4_Sem\Intelligent User Interface\Project\yolov8-dinov2-detection\runs\dino_inference\wsl_demo.jpg
```

On the first run, Torch Hub downloads the DINOv2 weights. Internet access is
required for this download and for any missing YOLO weights.

## Dynamic PyBullet ADAS demonstration

Run the trained detector in a dynamic PyBullet vehicle replay. The highest-
confidence detection controls steering, while PyBullet advances vehicle
dynamics and records position, speed, target, and steering telemetry:

```bash
python scripts/pybullet_adas_demo.py --device 0
```

For a visible WSLg/OpenCV simulator window, add `--gui`:

```bash
python scripts/pybullet_adas_demo.py --device 0 --gui
```

The `--gui` mode keeps PyBullet physics in headless mode and attempts to display
the live detector/vehicle telemetry through an OpenCV window. If the WSL Python
build has no GUI backend, it automatically continues and opens the generated
MP4 from Windows. Native PyBullet rendering remains available with
`--native-pybullet-gui` on systems where its OpenGL backend is stable.

From Windows, you can also double-click `launch_adas_gui.bat`, or run:

```powershell
.\launch_adas_gui.ps1 -Steps 180 -Device 0
```

The generated files are `runs/pybullet_adas/adas_replay.mp4` and
`runs/pybullet_adas/adas_replay.csv`. This is a closed-loop simulator
demonstration using a replayed camera frame. It is not a claim that the model
has been deployed to a physical vehicle; a real robot would additionally need
camera drivers, actuator limits, calibration, emergency stopping, and hardware
safety validation.

## nuScenes public dataset

nuScenes uses calibrated 3D annotations, so prepare its front-camera images and
projected 2D boxes before using YOLO training. Download the official
`v1.0-mini` or full dataset from https://www.nuscenes.org/nuscenes and extract
it into `data/nuscenes`.

Convert the `CAM_FRONT` annotations to a YOLO dataset:

```bash
python scripts/prepare_nuscenes.py \
  --root data/nuscenes \
  --version v1.0-mini \
  --output data/nuscenes_yolo \
  --limit 100
```

Remove `--limit 100` for all available samples. Inspect the generated
`data/nuscenes_yolo/data.yaml`, then train a detector with its class list:

```bash
python scripts/train.py \
  --data data/nuscenes_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --device 0
```

The COCO-trained checkpoint should not be presented as a nuScenes-trained
checkpoint: COCO and nuScenes have different category taxonomies. Use the
nuScenes checkpoint for nuScenes results, and report the dataset version,
number of images, classes, train/validation split, and projected-label
limitations in the assignment.

The `scripts/dino_inference.py` script is a sequential embedding demonstration.
For trained feature fusion, train the custom graph with `--fused`; its frozen
DINOv2 global embedding is projected to a compact context map and concatenated
with the deepest YOLO feature map. The context projector and YOLO detector are
trained together, while DINOv2 remains frozen.

## Dataset

The recommended development dataset is **COCO8**, an official small subset of
the standard COCO object-detection dataset. It is useful for verifying the
project on a laptop before moving to the full COCO dataset for final results.

Download it from the internet through Ultralytics:

```powershell
python scripts\download_dataset.py --dataset coco8
```

The ZIP is downloaded from the public Ultralytics assets release and extracted
locally into `data/coco8`. The ZIP is deleted after extraction.

For your own data, use this structure:

```text
data/custom/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

Each label line is:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to 0-1. Edit `configs/custom_data.yaml` with your classes and paths.

## Train

```powershell
python scripts\train.py --data data\coco8.yaml --epochs 10 --imgsz 640 --device cpu
```

Train YOLOv8 on the standard dataset:

```powershell
python scripts\train.py --baseline --data data\coco8.yaml --epochs 10 --device cpu
```

Run DINOv2 before YOLOv8 on GPU 0:

```powershell
python scripts\dino_inference.py `
  --input data\coco8\images\val\000000000036.jpg `
  --device 0
```

Evaluate both models:

```powershell
python scripts\evaluate.py --weights runs\baseline_yolov8\weights\best.pt --name baseline_metrics
python scripts\evaluate.py --weights runs\yolov8_training\weights\best.pt --name trained_metrics
```

For a custom dataset:

```powershell
python scripts\train.py --data configs\custom_data.yaml --epochs 50 --imgsz 640 --device 0
```

The default training model is `yolov8n.pt`. `dino_inference.py` is a separate
sequential embedding visualization; use `--fused` for DINO features connected
to the detector head.

Train the fused YOLOv8n + DINOv2 model from the pretrained YOLOv8n checkpoint:

```powershell
python scripts\train.py --fused --model yolov8n.pt --data data\coco8.yaml --epochs 10 --imgsz 320 --batch 4 --device 0
python scripts\evaluate.py --fused --weights runs\fused_yolov8\weights\best.pt --data data\coco8.yaml --name fused_metrics --device 0 --imgsz 640
```

Use the same dataset, train settings, validation split, and validation image
size for the baseline and fused model. Training uses full precision for both
models. The current COCO8 fused run scored zero on the validation metrics; see
`REPORT_TEMPLATE.md` for the comparison and do not interpret it as an
improvement. COCO8's four-image validation set is too small for generalization
claims.

## Important note

DINOv2 is loaded from Facebook Research through Torch Hub on first use. The
first run needs internet access and downloads pretrained weights. In fused
training, the DINOv2 backbone is frozen and its projected global feature map is
concatenated with YOLO features; only the projection and detector are trained.

## Expected deliverables

- Dataset description and class list.
- Baseline YOLOv8 results.
- YOLOv8 + DINOv2 results.
- Precision, recall, mAP50, and mAP50-95.
- Short comparison and limitations.
- Example prediction images or video.

Use `REPORT_TEMPLATE.md` for the assignment report. Create the FTP-ready
source archive with:

```powershell
python scripts\package_submission.py
```

The recorded baseline/fusion results and the prioritized follow-up experiments
are in `EXPERIMENT_RESULTS_AND_IMPROVEMENT_PLAN.md`.

For a self-contained Kaggle or Google Colab run on Pascal VOC, open
`YOLOv8_DINOv2_VOC_Kaggle_Colab.ipynb`. It explains the fusion changes, checks
the model shapes and adapter gradients, then proceeds directly to matched
baseline/fusion training and evaluation without a smoke-training run. The full
VOC download is about 2.8 GB. The notebook's optional Git section requires
setting `REPO_URL`; its token is prompted at runtime and is not stored in the
notebook.
