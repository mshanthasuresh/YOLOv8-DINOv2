# Speaker script

Generated together with the deck; the same text is in each slide's speaker notes.

## 01  /  TITLE

[25 s · finish by 0:25]

Good evening. At the midterm you asked me why LayerNorm plus GroupNorm, and I did not have a good answer. So I made that the research question.

I ran 45 controlled training runs to answer it. Short answer: fusion adds +10.5 mAP50-95 points, but the normalization does not explain it — the spatial DINOv2 features do.

## 02  /  STARTING POINT

[45 s · finish by 1:10]

This is what I showed at the midterm: one seed, ten epochs, and the test set also picked the checkpoint.

Token type, gate, normalization and initialization all changed at once, so the jump could not be attributed to anything.

While re-running I also found a weight-transfer bug: the pretrained neck filters were reading shifted channels. Fixing it is part of this work.

So I treated the midterm number as a hypothesis, not a result.

## 03  /  RESEARCH QUESTION

[40 s · finish by 1:50]

The question: does explicit normalization, rather than fusion itself, explain a stable gain — and is the gain worth a ViT's cost?

Four hypotheses, each with a pre-declared way to fail: H1 normalization, H2 fusion, H3 cost against a bigger CNN, H4 fewer labels and a far domain.

## 04  /  METHOD

[65 s · finish by 2:55]

Frozen DINOv2 ViT-S/14 patch tokens are projected to 64 channels and concatenated to YOLOv8n's P5 features right after SPPF.

The five arms change exactly one thing each: no norm, LayerNorm only, GroupNorm only, both, and the old gated design. Two controls come later.

Two fixes make the comparison fair: channel-correct weight transfer, and zero-initialized DINO columns — so at step zero every fused model is exactly the baseline.

A unit test checks that the output difference is zero. DINOv2 is never trained and never stored in the checkpoint.

## 05  /  PROTOCOL

[45 s · finish by 3:40]

Same data, schedule and starting weights for every arm. I train on 14,041 VOC images, choose the best epoch on the 2,510-image validation split, and score each model once on the 4,952-image test split.

The metric is mAP50-95: detection accuracy averaged over box-overlap thresholds from 0.5 to 0.95.

Every arm is trained with three seeds, and every difference I quote has a seed-paired bootstrap confidence interval.

## 06  /  WHY NORMALIZATION COULD MATTER

[50 s · finish by 4:30]

This is the intuition behind LN+GN. DINO tokens have RMS 2.5; YOLO's P5 only 0.32 — about 8× larger.

Depending on the normalization, the injected context starts between 0.015× and 0.147× of P5 — a 10× spread. ViT-S/14 has no outlier tokens, so the 'registers' argument does not apply here.

So the testable prediction is: if scale matters, these arms must end at different accuracies.

## 07  /  EXPERIMENT MAP

[50 s · finish by 5:20]

This is the depth of the study on one slide: 45 runs, 42 GPU-hours, and each row is a separate controlled question with its own runs.

Each row comes back later in the talk with its own figure and confidence interval.

The rest of the talk walks down the answer column.

## 08  /  MAIN RESULT

[70 s · finish by 6:30]

Held-out test, three seeds each: YOLOv8n 57.5, LN+GN 68.0 — a gain of +10.5 points, 95% CI +9.8 to +11.2.

Now H1: the four adapter variants land within a few tenths of each other. I don't just say 'not significant' — I test equivalence: the 90% CIs of LN, GN and LN+GN against no norm all lie inside ±0.5 points.

The only different variant is the gated one: +0.73 over no norm, Holm-corrected p < 0.003.

## 09  /  MECHANISM

[50 s · finish by 7:20]

Why doesn't a 10× scale difference matter? Every 50 iterations I logged how much of the first neck convolution's response comes from DINO channels.

It starts at zero by construction and ends between 0.32 and 0.60 depending on the arm — 1.9× apart — yet accuracy differs by only 0.26 points.

The network rescales the context itself, so normalization only changes the starting point.

## 10  /  CONTROLS

[60 s · finish by 8:20]

Two controls answer the 'why'. First: spatial layout, or just global scene information? Same LN+GN adapter, but the token map is averaged to one vector.

That gives +3.5 over the baseline; the spatial map adds +7.0 more — so most of the gain needs the layout.

Second: the gated design differs from no norm in two ways, a learnable gate and a 4× larger projector init. With the small init the gate alone gives +0.32; the larger init adds +0.41. Both contribute.

## 11  /  TRAINING BUDGET

[35 s · finish by 8:55]

One honest check: is the gain just faster learning? On the validation split the gap between LN+GN and YOLOv8n shrinks from +19.8 points at epoch 1 to +10.7 at epoch 15, and it is still shrinking.

So part of the gain is faster convergence, and a much longer schedule may narrow it. Every arm, and the YOLOv8s control, had the same 15-epoch budget.

## 12  /  WHERE IT HELPS

[40 s · finish by 9:35]

Where does it help? By object size, AP rises in every bin: small objects from 19.9 to 29.3, large ones from 63.4 to 75.5.

By class, all 20 classes improve. Animals gain the most — bird +19.3, dog +19.2, cat +14.9 — while person and car, which COCO-pretrained YOLO already knows well, gain the least.

## 13  /  ERROR ANALYSIS

[50 s · finish by 10:25]

Which errors does it fix? TIDE-style: fix one error type with an oracle and measure the AP50 recovered.

Wrong-class errors drop the most, missed objects next, box errors much less. At the 0.25 operating point, wrong-class detections fall from 626 to 277 and missed objects from 2240 to 1203.

So DINOv2 mainly fixes what an object is, more than where it is. The global-vector control cuts class errors to 0.99 but leaves box errors at 2.28 — better boxes need the spatial layout.

## 14  /  COST

[60 s · finish by 11:25]

Is it worth the cost? At 518 px the fused model needs 13× YOLOv8n's FLOPs, so the fair question is: why not just use a bigger CNN?

YOLOv8s, trained the same way, gets 61.8. Fusion at 518 beats it by +6.3.

At 224 px fusion needs fewer FLOPs than YOLOv8s and is faster per image in batches — and still wins by +3.0. So H3 holds.

At batch 1 the GPU mostly waits for the CPU to launch kernels, so I replay the whole forward pass as one CUDA graph: same kernels, bit-identical outputs, and LN+GN at 518 drops from 19 to 10 ms, at 224 from 19 to 5 ms.

## 15  /  GENERALIZATION

[50 s · finish by 12:15]

H4. With only 10% of the labels the gain grows to +15.0: the fused model with 10% labels (57.2) is about as good as the baseline with all labels (57.5).

On VisDrone — aerial images, tiny objects — the gain shrinks to +0.8, as predicted: DINO patches are 14 px and the context enters at stride 32.

Normalization again makes no difference there.

## 16  /  EXAMPLES

[35 s · finish by 12:50]

Examples, chosen automatically by the largest per-image change between the two models. Fusion improves 1517 images and hurts 782 of 4695.

The right panel shows where it hurts — failures are included on purpose.

## 17  /  VALIDITY

[40 s · finish by 13:30]

Finally, the boring explanations I ruled out. Run-to-run noise: three seeds and bootstrap confidence intervals. Choosing on the test set: checkpoints are chosen on validation.

Confounds: one factor per arm, plus two controls. 'Not significant' versus 'the same': an equivalence test with Holm correction. Bugs: 19 unit tests.

Extra compute: the YOLOv8s control. One dataset: VisDrone and the 10%-label regime. What remains: three seeds, one ViT size, and the 15-epoch budget.

## 18  /  CONCLUSIONS

[50 s · finish by 14:20]

To conclude. H1 is rejected: normalization changes the injected scale ten-fold, but not the accuracy. H2 holds: +10.5 points.

H3 holds: fusion beats a bigger CNN even at lower compute. H4 holds: a larger gain with fewer labels, a smaller one on tiny aerial objects.

What actually matters is the spatial DINOv2 context; a learnable gate adds a little.

Main limitation: the 15-epoch budget — the dev gap is still shrinking, so I cannot claim it persists at full convergence. Thank you, I'm happy to take questions.

## B1  /  BACKUP · RELATED WORK & GAP

[Backup] Use if asked how this differs from ViT-Adapter or ViTDet: they make the ViT the backbone; here a frozen ViT is side context for a pretrained CNN, and each normalization is isolated.

## B2  /  BACKUP · REPRODUCIBILITY

Q: Why LayerNorm + GroupNorm, why not BatchNorm?

A: The ViT tokens and the CNN's P5 features have different scales; normalization decides how strongly DINOv2 speaks. At initialization the GN arms inject 0.147× the P5 scale versus 0.036× without norm and 0.015× with LN only.

GroupNorm does not depend on batch size (batch 16, and batch 1 at deployment) and adds no running statistics. ViT-S/14 shows no

high-norm outlier tokens, so the 'registers' argument is not the reason here — the scale gap is.



Q: Is the gain bigger than the noise?

A: LN+GN vs YOLOv8n: +10.5 mAP50-95 (3 seeds, 95% CI [+9.79, +11.17] — beyond noise).



Q: Is it worth the latency?

A: See the YOLOv8s control and the DINO-resolution sweep on the cost slide.



Q: How do you know the copied pretrained filters read the right channels?

A: Unit test: with DINO columns zeroed the fused model's output equals the baseline's exactly (max |diff| = 0) for every arm,

and a second test silences the DINO branch after randomizing its columns.



Q: How is this different from the IUI project?

A: IUI produced one confounded, single-seed run. Here: isolated arms, fixed weight transfer, dev/test separation, 3 seeds,

bootstrap CIs, mechanism logging, cost and generalization experiments.
