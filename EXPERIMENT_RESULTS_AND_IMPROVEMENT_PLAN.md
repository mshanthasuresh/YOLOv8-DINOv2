# YOLOv8 + DINOv2 Experiment Results and Improvement Plan

Date recorded: 2026-09-27

## Summary

The earlier COCO8 experiment used a zero-initialized DINO projector and failed: the trained fused checkpoint kept zero projection weights/bias and scored zero on validation. The source was then updated to use ImageNet normalization, a small nonzero projector/context initialization, and adapter-norm logging. A larger Pascal VOC experiment was run on Kaggle with a GPU. The fused model trained and its adapter norms changed during training, but the fusion model scored **lower** than the baseline on all four metrics. DINOv2 did not improve detection in this experiment.

## VOC Results (Kaggle, 2026-09-29)

Both models used the same Pascal VOC data, 640px images, batch size 2, 10 epochs, AdamW optimizer, learning rate 0.001, FP32, seed 42, and VOC2007 test images as the validation holdout. The DINO adapter was observed changing during training (epoch 4: weight_norm=1.08746, bias_norm=2.62007; epoch 9: weight_norm=1.23016, bias_norm=2.63433).

| Model | Precision | Recall | mAP50 | mAP50-95 | Inference ms/img |
|---|---:|---:|---:|---:|---:|
| YOLOv8n baseline | 0.6768 | 0.6427 | 0.6852 | 0.4768 | 2.68 |
| YOLOv8n + DINOv2 fusion | 0.4617 | 0.3911 | 0.3691 | 0.2182 | 36.88 |

The fusion model is worse than the baseline on every metric and 13.8x slower per image. The adapter learned nonzero parameters, but the added global-context pathway did not help detection and may have interfered with the pretrained detector. Full output artifacts will be available for download once Kaggle finalizes the run.

## Earlier COCO8 Results (Failed Run — Kept for Provenance)

Both fine-tuned runs used COCO8, the same four training images and four validation images, 10 epochs, image size 320 for training, batch size 4, GPU 0, full-precision training, and image size 640 for final validation.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Fine-tuned YOLOv8n baseline | 0.6012 | 0.9023 | 0.8873 | 0.6253 |
| Earlier YOLOv8n + DINOv2 (zero-init, failed) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The baseline is in `runs/baseline_fp32_same_split_640/metrics.json`; its checkpoint is `runs/baseline_yolov8-2/weights/best.pt`. The fused metrics are in `runs/fused_metrics_fp32/metrics.json`; its checkpoint is `runs/fused_yolov8-3/weights/best.pt`. Per-epoch training metrics are in `runs/baseline_yolov8-2/results.csv` and `runs/fused_yolov8-3/results.csv`.

The zero-initialized fused graph, before training, scored P=0.6194, R=0.8333, mAP50=0.8875, and mAP50-95=0.6291, matching the pretrained `yolov8n.pt` checkpoint on the same validation images. A same-input forward comparison also showed the detector outputs match to numerical precision before training.

## Verified Observations

- DINOv2 is connected to the detector graph and the graph produces valid YOLO prediction tensors.
- DINOv2 remains frozen in the saved checkpoint.
- The trained fusion checkpoint has zero-valued DINO projection weights and bias. The adapter therefore did not learn a nonzero mapping in this run.
- The fused model's validation metrics are zero at both the 320px training-time validation and explicit 640px evaluation.
- COCO8 has only four training and four validation images here. These scores are smoke-test results and cannot support generalization claims.
- The failed run used mean/std 0.5 for DINO input normalization. The current source and cloud notebook now use ImageNet mean/std; this correction is not yet represented in any completed training metrics.
- A one-image detector-loss probe produced a nonzero projector gradient, and a direct AdamW step changed the projector. The previous full training run still saved a zero projector; the notebook adds per-epoch adapter-norm logging to expose this in a larger run.
- The current DINO branch returns a global class embedding and broadcasts it spatially. It does not provide spatially varying DINO patch features to the detector.

The exact reason the projector stayed zero has not yet been isolated. The recommendations below are hypotheses to test in order, not claims that the root cause is already known.

## Ranked Improvement Plan

### 1. Monitor adapter learning during training

Before a full run, execute one real detection-loss backward/optimizer step and log:

- DINO projector weight and bias gradient norms.
- The gradient norm for the YOLO convolution weights connected to the new context channels.
- Whether the projector parameters are in the optimizer parameter groups.
- Projector and context-channel weight norms before and after `optimizer.step()`.

The direct one-step check now exists in `YOLOv8_DINOv2_VOC_Kaggle_Colab.ipynb`. Fail fast if any required gradient is `None`, non-finite, or the adapter parameters do not change. The notebook also prints adapter norms per epoch; a full run must confirm these continue to change.

### 2. Remove the zero-signal initialization bottleneck

The failed run zero-initialized the projector and used very small new context-channel weights. The current source and notebook use a nonzero projector and slightly stronger context-channel initialization while preserving pretrained YOLO filters. A residual/gated fusion path remains a follow-up ablation if the revised path still fails to learn.

Run a one-batch overfit test first. The model should drive training loss down and produce nonzero detections on the same tiny training images before spending time on a benchmark run.

### 3. Match DINOv2 preprocessing to pretraining

Use RGB inputs in [0, 1] with ImageNet mean/std. The current source and notebook apply this normalization. Keep training and inference preprocessing identical and inspect embedding/context ranges during the next run.

### 4. Stabilize fine-tuning after the adapter works

First train only the projection/fusion components for a short warm-up. Then unfreeze the YOLO backbone/head with a lower learning rate than the fusion adapter. Keep the DINOv2 backbone frozen initially. Monitor per-component gradient norms and compare the trained checkpoint against its initialization after every epoch.

### 5. Use spatial DINO features as a separate ablation

Once the global-context version trains correctly, test DINOv2 patch-token features reshaped to their patch grid and resized to the YOLO feature scales. Detection needs localization; a single class token repeated over every spatial position has scene-level semantics but no location-specific DINO detail. Compare global-token and patch-token variants separately rather than changing both at once.

### 6. Replace COCO8 for the final comparison

Keep COCO8 for fast graph and gradient smoke tests only. `YOLOv8_DINOv2_VOC_Kaggle_Colab.ipynb` is prepared for Pascal VOC (16,551 training images and 4,952 VOC2007 test images used as the comparison holdout). The VOC run has not yet been executed. Report these split semantics, class distribution, and limitations; repeat training with multiple seeds if time permits.

### 7. Keep evaluation controlled

For every run, hold constant dataset split, input size, validation settings, and metric implementation. Report precision, recall, mAP50, mAP50-95, inference latency, and parameter/FLOP cost. Save prediction plots and confusion matrices. Compare pretrained YOLO, fine-tuned YOLO, and fine-tuned fusion; do not attribute an accuracy change to DINO if its adapter did not update.

## Recommended Next Experiment

1. Run the notebook's one-epoch 1% VOC smoke test and confirm adapter norms change.
2. If the smoke test passes, train baseline and fusion with identical full-VOC settings.
3. Evaluate both checkpoints at the same image size and add the results to the table above.
4. Treat VOC2007 test as the comparison holdout; avoid repeatedly tuning on it.

## References

- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/): pretrained initialization, learning-rate/optimizer configuration, and validation during training.
- [Ultralytics validation guide](https://docs.ultralytics.com/modes/val/): controlled validation, detection metrics, prediction plots, and exported results.
- [Official DINOv2 repository](https://github.com/facebookresearch/dinov2): pretrained backbones, image- and pixel-level features, and dense-task examples.
- [DINOv2 paper](https://arxiv.org/abs/2304.07193): pretrained visual features and image/pixel-level evaluation.
- [PyTorch automatic mixed precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html): gradient scaling prevents small FP16 gradients from underflowing; the current runs used FP32, so AMP is not the explanation for this recorded failure.
