# Final presentation: rehearsal guide (Friday)

**Graded on:** depth of experimentation and quality of analysis.
**Plan for:** a 15-minute talk. The script runs 14:20, leaving about 40 seconds of slack.
**Ask the coordinator beforehand:** are questions inside the 15 minutes? If yes, use the 12-minute cut in section 2.
**Deck:** 18 talk slides, then 2 backup slides (B1–B2) that you show only if someone asks.
**Speaker notes:** every slide's notes start with its time budget and the clock time to finish by, for example *[45 s · finish by 1:10]*. In PowerPoint use Slide Show, then *Use Presenter View*.

---

## 1. The 30-second answer (memorize this)

> At the midterm I was asked why LayerNorm plus GroupNorm, and I couldn't answer. So I made it the research question and ran 45 controlled runs.
>
> Fusing frozen DINOv2 into YOLOv8n adds **+10.5 mAP50-95** on held-out VOC07 test. The normalization does **not** explain it: LN, GN and LN+GN are statistically equivalent to no normalization.
>
> What matters is the **spatial** DINOv2 context, which gives two-thirds of the gain. It mainly fixes **wrong-class** errors. It beats a bigger CNN (YOLOv8s) even at **lower compute**. The gain grows to **+15** with 10% of the labels and shrinks to **+0.8** on tiny aerial objects.

---

## 2. Talk plan: 18 slides in about 14 minutes

| # | Slide | Time | Finish by | The one message |
|---|---|---|---|---|
| 1 | Title | 0:25 | 0:25 | You asked "why LN+GN"; here is the answer, backed by 45 runs. |
| 2 | Starting point | 0:45 | 1:10 | The midterm jump changed five things at once, plus a weight-transfer bug. It was a hypothesis, not a result. |
| 3 | Research question | 0:40 | 1:50 | Four hypotheses, each with a stated way to fail. |
| 4 | Method | 1:05 | 2:55 | One factor per arm. Channel-correct transfer and zero-init mean every arm starts as exactly the baseline. |
| 5 | Protocol | 0:45 | 3:40 | Same data and schedule for every arm. Checkpoint chosen on validation, test scored once. mAP50-95, 3 seeds, bootstrap CIs. |
| 6 | Why normalization could matter | 0:50 | 4:30 | The intuition: an 8× scale gap and a 10× spread at init. Prediction: if scale matters, accuracies differ. |
| 7 | Experiment map | 0:50 | 5:20 | Depth: 45 runs, 42 GPU-hours, every row a controlled question. |
| 8 | Main result | 1:10 | 6:30 | +10.5 [9.8, 11.2]. Normalization arms are equivalent within ±0.5. Only gating differs (+0.73). |
| 9 | Mechanism | 0:50 | 7:20 | Reliance on DINO differs 1.9× across arms, accuracy by 0.26: the network rescales the context itself. |
| 10 | Controls | 1:00 | 8:20 | Spatial layout gives two-thirds of the gain. Gate and init each add about 0.3–0.4. |
| 11 | Training budget | 0:35 | 8:55 | The gap shrinks from 19.8 to 10.7 over 15 epochs: part of the gain is faster learning. Say this honestly. |
| 12 | Where it helps | 0:40 | 9:35 | Every object size and all 20 classes improve; animals most, person and car least. |
| 13 | Error analysis | 0:50 | 10:25 | It fixes what an object is (wrong class −74%) more than where it is (boxes −31%). |
| 14 | Cost | 1:00 | 11:25 | It beats YOLOv8s by +6.3. At 224 px it uses fewer FLOPs than YOLOv8s and still wins by +3.0. |
| 15 | Generalization | 0:50 | 12:15 | +15.0 with 10% of labels; +0.8 on VisDrone; normalization irrelevant on both. |
| 16 | Examples | 0:35 | 12:50 | Where it helps and where it hurts; failures are shown on purpose. |
| 17 | Threats to validity | 0:40 | 13:30 | Every boring explanation and the experiment that rules it out. |
| 18 | Conclusions | 0:50 | 14:20 | Verdicts: H1 rejected, H2–H4 supported. Limitation: the 15-epoch budget. |

**Checkpoints during the talk:**
- Main result (slide 8) should start at about **5:20**.
- Cost (slide 14) should start at about **10:25**.
- If you are more than 30 seconds behind, shorten Examples (16) and Threats (17) to one sentence each.

**If questions are inside the 15 minutes (a 12-minute talk):**
1. Skip Protocol (5) and say its key sentence on the Experiment map instead: "checkpoint chosen on validation, test scored once, three seeds".
2. Skip Where it helps (12).
3. Skip Threats (17).
4. Cut Training budget (11) to one sentence in the Conclusions.

That saves about 2:40, so the talk ends near 11:40.

**Backup slides** (the slide number to type during the show is in brackets):
- B1 Related work [19]
- B2 Reproducibility and likely questions [20]

Jump to one only when a question needs it.

---

## 3. Numbers to know by heart

| Topic | Number |
|---|---|
| Headline (VOC07 test mAP50-95, 3 seeds) | YOLOv8n **57.5** → LN+GN **68.0**, Δ **+10.5** [9.8, 11.2] |
| mAP50 | 79.1 → 88.9 |
| Normalization vs no norm (90% CIs) | LN −0.16 [−0.33, +0.02]; GN +0.10 [−0.06, +0.26]; LN+GN +0.05 [−0.13, +0.23]; all inside ±0.5 |
| Gated vs no norm | **+0.73** [0.52, 0.92], Holm p < 0.003 |
| Scale at initialization | DINO RMS 2.5 vs P5 0.32 (about 8×); injected 0.015× to 0.147× of P5 (10× spread) |
| DINO share of the first neck conv | ends at 0.32 (GN arms) to 0.60 (LN only); accuracy spread only 0.26 |
| Global-vector control | +3.5 over baseline; spatial map adds **+7.0** [6.5, 7.5], i.e. two-thirds of the gain |
| Gate vs init | gate alone +0.32 [0.11, 0.52]; larger init +0.41 [0.17, 0.59] |
| Error analysis: AP50 lost (baseline → LN+GN) | wrong class 2.03 → 0.52 (−74%); missed 0.45 → 0.14 (−69%); poor box 2.19 → 1.52 (−31%); background 4.40 → 3.48 (−21%) |
| Errors at conf ≥ 0.25 | wrong-class detections 626 → 277; missed objects 2,240 → 1,203 of 12,032 |
| YOLOv8s | **61.8**; fusion @518 is +6.3 [5.6, 6.9]; fusion @224 is **+3.0** [2.3, 3.5] |
| Cost: YOLOv8n | 8.1 GFLOPs, 1.65 ms/img |
| Cost: YOLOv8s | 28.5 GFLOPs, 3.69 ms/img |
| Cost: fusion @224 | 20.4 GFLOPs, 3.20 ms/img |
| Cost: fusion @518 | 101.6 GFLOPs, 9.65 ms/img |
| DINOv2 resolution | 224: 64.6; 336: 66.7; 448: 67.9; 518: 68.0 |
| 10% labels | 42.2 → 57.2, **+15.0** [14.3, 15.7]; fusion with 10% labels ≈ baseline with 100% (57.5) |
| VisDrone | 12.5 → 13.2, **+0.8** [0.6, 0.9]; LN+GN vs no norm +0.04 [−0.06, +0.15] |
| Object size (AP) | small 19.9 → 29.3; medium 40.4 → 48.8; large 63.4 → 75.5 |
| Per class | every class improves; most: bird +19.3, dog +19.2, cat +14.9; least: car +4.0, person +5.6 |
| Dev gap (training dynamics) | 19.8 at epoch 1 → 13.3 at epoch 10 → 10.7 at epoch 15, still shrinking |
| Scale of the study | 45 runs, 42 GPU-hours, 17 unit tests |

---

## 4. Likely questions and answers

Keep every answer under 30 seconds: say the claim, then the number, then stop.

**Method and intuition**

1. **"So why did you use LayerNorm and GroupNorm?"** (The TA asked this at the midterm.)
   The intuition was scale matching. DINO tokens are about 8× larger than YOLO's P5 features, and normalization sets how loudly DINO speaks: a 10× spread at initialization. I tested that intuition directly. Normalization changes the starting scale and how much the detector relies on DINO, but final accuracy is equivalent within ±0.5 points. Zero-init plus training does the scaling. So LN+GN is not what makes fusion work.

2. **"Why inject only at P5?"**
   P5 is coarse (stride 32), which matches DINO's 14-px patch grid. It costs a single concatenation. Injecting at P3/P4 might help small objects, and the small VisDrone gain suggests that. It is my first piece of future work.

3. **"Why frozen DINOv2, and why ViT-S?"**
   Frozen keeps the comparison clean: only the adapter and YOLO train, and there is no forgetting. ViT-S is the cheapest. Larger ViTs are where the "registers" outlier tokens appear, so normalization might matter more there. That is future work.

4. **"What is the DINO share?"**
   I split the first neck convolution into its DINO part and its YOLO part. The share is RMS(DINO response) / (RMS(DINO) + RMS(YOLO)). It starts at exactly 0 because of zero-init, and I logged it every 50 iterations.

5. **"How do you know the copied weights are right?"**
   There is a unit test: with the DINO columns zeroed, the fused model's output equals the baseline exactly (max difference 0) for every arm. In total, 17 tests cover offsets, pickling, resume, gradient flow and the error analysis.

**Statistics**

6. **"How did you compute the confidence intervals?"**
   With a paired image bootstrap. I resample the 4,952 test images with replacement 400 times. Both models are re-scored on the same resample, and the difference is averaged over seed-matched pairs. My AP code reproduces Ultralytics' numbers exactly.

7. **"Why a ±0.5 margin for equivalence?"**
   It is about the baseline's seed-to-seed standard deviation (0.4). A smaller difference is below run-to-run noise. All 90% CIs actually fall within ±0.33, so the conclusion survives a much tighter margin.

8. **"Only three seeds?"**
   Yes: three per arm, and two for VisDrone and the resolution sweep. The seed standard deviation is 0.06–0.4, far below the effects I claim. Seeds cover training noise and the bootstrap covers test-set sampling. More seeds would tighten the gate result most.

9. **"Multiple comparisons?"**
   Holm correction across the four normalization comparisons. The gated result survives (p < 0.003); the others are equivalent anyway.

**Results**

10. **"Is it just faster convergence?"** (Slide 11.)
    Partly, and I say so. The dev gap shrinks from about 20 to 10.7 points over 15 epochs and is still shrinking. All arms and YOLOv8s got the same budget. I can't claim the gap persists at full convergence; that is my main limitation.

11. **"Why does DINOv2 help classification more than localization?"**
    DINOv2 features are semantic and coarse: 14-px patches, injected at stride 32. Box precision comes from YOLO's high-resolution features. The error analysis shows exactly this: wrong-class errors −74%, box errors −31%. The global-vector control halves class errors but does nothing for boxes, so better boxes need the spatial layout.

12. **"Why does the gate help when normalization doesn't?"**
    The gate is a learnable per-channel scale, so the model can pick useful DINO channels. It is worth +0.32 on its own, and the larger init adds +0.41. Both are small next to the +10.5 from spatial context.

13. **"Which classes benefit?"** (Slide 12.)
    All 20 classes improve. Animals gain most: bird +19, dog +19, cat +15. Person and car gain least (+5.6 and +4.0), because COCO-pretrained YOLO already knows them well.

**Cost and fairness**

14. **"Isn't it just a bigger model?"**
    I controlled for that. YOLOv8s has 3.5× YOLOv8n's FLOPs and gets 61.8. Fusion at 224 px uses fewer FLOPs than YOLOv8s (20 vs 28) and less batched time (3.2 vs 3.7 ms), and is still +3.0 better.

15. **"Was the YOLOv8s comparison fair?"**
    Yes: same data, 15-epoch schedule, optimizer, COCO initialization, checkpoint selection on dev, and 3 seeds.

16. **"Batch-1 latency is worse. Isn't that a problem?"**
    Yes for real-time single-image use: 22.6 vs 9.5 ms on this laptop GPU. That figure is launch-bound, which is why I also report FLOPs and batched time. I show it in the table rather than hide it.

17. **"Did you tune hyperparameters per arm?"**
    No. Every arm uses identical settings, which avoids tuning bias. The risk is that some arm would prefer a different learning rate; that is a stated limitation.

**Generalization**

18. **"Why is the VisDrone gain so small?"**
    The objects are tiny and the images are 1,360–2,000 px wide. After resizing, each DINO patch covers a large region, and the context enters at stride 32. DINOv2 was also trained on ground-level photos, not aerial views. It still helps (+0.8, CI excludes 0), and normalization again makes no difference.

19. **"Could DINOv2 have seen VOC test images in pretraining?"**
    DINOv2's authors removed near-duplicates of their evaluation benchmarks from the pretraining data, but I can't fully rule out overlap with VOC07 test. DINOv2 never saw labels, and the same pattern holds with 10% of the labels and on VisDrone.

20. **"What would you do next?"**
    - Train to convergence, to settle the convergence question.
    - Inject at P3/P4 for small objects.
    - Try ViT-B with registers.
    - Distill DINOv2 into YOLO to remove the inference cost.

**If you don't know an answer:**
> "I didn't test that directly. My closest evidence is ___. I would test it by ___."

That is better than guessing; it shows analytical thinking.

---

## 5. Delivery tips

- **Open with the midterm question.** It shows you took the feedback seriously and frames the whole talk.
- **Say "statistically equivalent within ±0.5", never "no difference".** Absence of significance is not evidence of absence; the equivalence test is.
- **Give every number its context once,** for example "+10.5, confidence interval 9.8 to 11.2, three seeds". After that use plain words.
- **Point at the figure** element you are talking about, for example "the teal bar" or "this row".
- **State the limitation yourself** (the 15-epoch budget). Examiners trust a speaker who names their own weak spot.
- **Don't read the slides.** The notes in Presenter View are a safety net.

---

## 6. Rehearsal plan

| When | What |
|---|---|
| Wednesday evening | Read the speaker notes aloud twice with a timer. Mark the slides where you run over. |
| Thursday | Three timed full runs; target 14:00–14:30. Record one on your phone and watch it once. Say answers 1, 6, 7, 10, 11 and 14 aloud until each takes under 30 seconds. |
| Thursday night | Mock Q&A with me: reply **"mock Q&A"** and I'll ask TA-style questions one at a time and critique your answers. |
| Friday morning | One final run. Check screen sharing with Presenter View. Open the deck file in advance. Know how to jump to backup slides (type the slide number, then press Enter). |
