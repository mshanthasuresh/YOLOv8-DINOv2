# Midterm Report: YOLOv8 + DINOv2 Object Detection

## 1. Objective

This project compares pretrained YOLOv8n with a YOLOv8n detector augmented by
a frozen DINOv2 global image embedding. The goal is to test whether injecting
global visual context into the detector improves object-detection metrics.

## 2. Dataset

- Dataset: COCO8, the official eight-image COCO development subset
- Source: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
- Classes: 80 COCO categories are defined in the annotation format
- Split used: 4 training images and 4 validation images
- Classes present in validation: person, dog, horse, elephant, umbrella, and potted plant
- Purpose: COCO8 is small enough for a local smoke test; it is not large enough
	to support a statistically strong generalization claim.

The dataset is public and already downloaded under `data/coco8`. The corrected
dataset configuration is `data/coco8.yaml`.

## 3. Method

The fused graph preserves the raw image through `ConvDummy`, runs the image
through frozen DINOv2 ViT-S/14, projects its 384-dimensional global embedding
to a three-channel context map, and broadcasts that map to the deepest YOLO
feature-map size. The context map is concatenated with the SPPF feature before
the YOLO detection head. The context projector and detector are trainable;
DINOv2 stays frozen. The compact three-channel projection is an implementation
choice for the current Ultralytics parser and differs from the 1024-channel
projection in the reference design.

The fused model starts from pretrained `yolov8n.pt` weights. Compatible YOLO
weights are transferred across the shifted YAML layer indices, and the two
widened C2f input convolutions are padded to retain pretrained weights around
the added context channels. In the recorded COCO8 run, the DINO projector started
at zero so the initial fused detector reproduced pretrained YOLO before
fine-tuning. After that failed run, the source was updated to use a small
nonzero projector initialization and ImageNet normalization. The updated code
has not yet been trained or evaluated; its results must not be confused with
the completed COCO8 results below.

Both the baseline and fused model were trained for 10 epochs at image size 320,
batch size 4, on the same four COCO8 training images, with GPU 0 and AMP
disabled. Both were evaluated at image size 640 on the same four validation
images. Commands used:

```powershell
python scripts\train.py --baseline --model yolov8n.pt --data data\coco8.yaml --epochs 10 --imgsz 320 --batch 4 --device 0
python scripts\train.py --fused --model yolov8n.pt --data data\coco8.yaml --epochs 10 --imgsz 320 --batch 4 --device 0
python scripts\evaluate.py --weights runs\baseline_yolov8-2\weights\best.pt --name baseline_fp32_same_split_640 --device 0 --imgsz 640
python scripts\evaluate.py --fused --weights runs\fused_yolov8-3\weights\best.pt --name fused_metrics_fp32 --device 0 --imgsz 640
```

## 4. Baseline Results

| Metric | Pretrained YOLOv8n | Fine-tuned YOLOv8n | Fine-tuned YOLOv8n + DINOv2 |
|---|---:|---:|---:|
| Precision | 0.619 | 0.601 | 0.000 |
| Recall | 0.833 | 0.902 | 0.000 |
| mAP50 | 0.888 | 0.887 | 0.000 |
| mAP50-95 | 0.629 | 0.625 | 0.000 |

The fused checkpoint produced no true-positive detections on the validation
set. It therefore does not show an accuracy improvement; this run is a failed
fusion-training result and must not be presented as evidence that DINOv2
improves YOLO. The initial fused graph, before training, reproduced the
pretrained YOLO metrics, which verifies the feature routing and pretrained
weight transfer. The failure occurs during fused fine-tuning and still needs
investigation.

The sequential `scripts/dino_inference.py` demonstration remains separate from
the fused model. It extracts and displays a DINO embedding but does not affect
the detector's predictions.

## 6. Utility of the Pretrained Model

similarity search, and as input to a future detector-fusion or re-ranking
DINOv2 features were structurally connected to the YOLO detection head, but
the trained checkpoint failed to detect the validation objects. The result
shows why the baseline comparison is necessary: adding a pretrained feature
extractor is not itself evidence of better detection. The zero metrics call for
debugging and retraining before any claim about the utility of DINOv2. A
Kaggle/Colab Pascal VOC notebook is prepared for that next experiment, but no
larger-dataset result is available yet.

## 7. Limitations

- COCO8 contains only four training and four validation images, so the metrics
	are a smoke-test result rather than a statistically reliable benchmark.
- DINOv2 adds substantial startup and inference cost compared with YOLO alone.
- The DINO projection remained ineffective in the trained run, and fused
	fine-tuning collapsed to zero validation detections.
- The pretrained models may suffer from domain shift on non-COCO images.
- The experiment uses a single short training run and no hyperparameter search.
- Repeating the experiment on full COCO or a larger validation set is required
	before claiming improved accuracy or robustness.

## 8. Conclusion

The fused architecture was implemented, trained, and evaluated against a
fine-tuned YOLOv8n baseline using the same COCO8 split and metrics. The baseline
achieved 0.601 precision, 0.902 recall, 0.887 mAP50, and 0.625 mAP50-95. The
fused model scored zero on all four metrics, so the assignment comparison is
complete as an experiment but does not support the hypothesis that DINOv2
improves detection. The training collapse must be fixed and the comparison
repeated on a larger dataset before drawing a performance conclusion.

## 9. Dynamic ADAS Demonstration

To demonstrate how the trained detector can participate in an interactive
system, a PyBullet replay was added. The detector processes the camera frame,
selects the highest-confidence object, and converts its horizontal image offset
into a steering command. PyBullet advances a dynamic racecar and records
vehicle position, speed, target class, confidence, and steering.

Run it with:

```bash
python scripts/pybullet_adas_demo.py --device 0
```

The demonstration produces `runs/pybullet_adas/adas_replay.mp4` and
`runs/pybullet_adas/adas_replay.csv`. This is a simulator integration, not a
real-vehicle deployment. A physical robot or ADAS car would require calibrated
cameras, actuator interfaces, safety limits, collision checking, emergency
stopping, and hardware-in-the-loop validation.

## 10. nuScenes Extension

nuScenes is a more appropriate public autonomous-driving dataset for the next
experiment. Its `CAM_FRONT` images and calibrated 3D annotations are converted
to YOLO-format 2D labels by projecting visible 3D box corners with the camera
intrinsic matrix. The conversion script is:

```bash
python scripts/prepare_nuscenes.py --root data/nuscenes \
	--version v1.0-mini --output data/nuscenes_yolo --limit 100
```

The `--limit 100` option is only for a smoke test. The full dataset should be
used for the final experiment after downloading it from the official nuScenes
site. Results must be reported separately from COCO8 because the class
taxonomy, camera calibration, scene distribution, and annotation format differ.
