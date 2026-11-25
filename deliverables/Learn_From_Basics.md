# Your project, from the basics

**How to use this file:**
- Read one part at a time, in order.
- After each part, close the file and say the **"Say it"** lines aloud from memory.
- Then open the slides and find where that idea appears.
- The timed script for every slide is in the deck's speaker notes. It will make sense once you know Parts 0–4.

**Suggested plan:**
- Today: Parts 0–2.
- Tomorrow: Parts 3–4, the self-check (Part 6) and two timed run-throughs.
- Friday: one final run-through.

---

## Part 0: The whole story in 6 sentences

1. **YOLOv8n** is a small, fast object detector.
2. **DINOv2** is a large vision model that learned very good image features from 142 million images, *without any labels*.
3. I **added DINOv2's features into YOLOv8n**. This is called *fusion*.
4. At the midterm my fusion used two normalization layers (**LayerNorm + GroupNorm**), and the TA asked: *"Why?"*
5. I tested this properly with **45 training runs**. Fusion improves accuracy a lot (**+10.5 points**), but the normalization layers are **not** the reason.
6. The real reason is DINOv2's **spatial information**: *what* is *where* in the image. The cost is acceptable, because even a cheap version beats a bigger YOLO.

> **Say it:** "Fusion adds 10.5 points. Normalization is not the reason. DINOv2's spatial features are."

---

## Part 1: The basic ideas

### 1.1 Object detection
- **Input:** an image.
- **Output:** for every object, a **box**, a **class label** (dog, car, person…) and a **confidence score** (how sure the model is).
- A detector can make two main kinds of mistakes:
  - **wrong label**: it says "cat" for a dog (a *classification* error);
  - **wrong box position**: the box is shifted or the wrong size (a *localization* error).

> **Say it:** "Object detection means finding each object, drawing a box around it and naming it."

### 1.2 YOLOv8n, the detector
- **YOLO** = *You Only Look Once*. One pass through a neural network gives all the boxes, so it is fast.
- **v8** = version 8 (from the company Ultralytics).
- **n** = *nano*: the smallest version, about 3 million parameters. **s** = *small*: about 11 million parameters.
- It has three parts:

```
image ──► BACKBONE ──────────► NECK ───────────────► HEAD
          (extracts features)   (mixes big & small    (predicts boxes,
                                 feature maps)         classes, scores)
```

- The backbone produces **feature maps** at three sizes:
  - **P3** (fine detail, for small objects);
  - **P4**;
  - **P5** (coarse, a 20×20 grid for a 640×640 image, for big objects and overall context).
- **SPPF** is the last block of the backbone, and P5 is its output. A **channel** is one "layer" of a feature map; P5 has 256 channels.
- I start from weights already trained on **COCO** (a big detection dataset) and **fine-tune** them on my data.

> **Say it:** "YOLOv8n is the smallest YOLO. I start from COCO-pretrained weights and fine-tune it."

### 1.3 DINOv2, the extra "knowledge"
- DINOv2 is a **Vision Transformer (ViT)** from Meta.
- It was trained with **self-supervised learning**: it learned from 142 million images **without labels**, by comparing different views of the same image.
- It cuts the image into **14×14-pixel patches**. For each patch it outputs a **token**: a vector of **384 numbers** describing that patch.
  - With a 518×518 input there are 37×37 = 1,369 patches.
- These tokens describe **what things are**. For example, all patches on a dog get similar vectors.
- **ViT-S/14** means the *small* ViT with *14*-pixel patches.
- **Frozen** means I never change DINOv2's weights. I only use its output.

> **Say it:** "DINOv2 is a self-supervised vision transformer. I use it frozen, as a source of rich features."

### 1.4 Fusion: how the two are combined

```
image ─┬─► YOLOv8n backbone ──► P5: 256 channels, 20×20 ─────────────┐
       │                                                              ├─► concat: 320 channels ─► YOLO neck + head ─► boxes
       └─► DINOv2 (frozen) ──► tokens: 384 numbers, 37×37 ─► ADAPTER ─┘
                                                     (to 64 channels, resized to 20×20)
```

- The **adapter** is a small set of layers. It turns DINOv2's 384 numbers per patch into **64 channels** and resizes the 37×37 grid to **20×20**, so the result lines up with P5.
- **Concatenate** means placing the two feature maps side by side: 256 + 64 = **320 channels**.

> **Say it:** "I project DINOv2's patch features to 64 channels and concatenate them to YOLO's P5 features, right after SPPF."

### 1.5 Normalization: the question of my project
- **The problem:** the two streams have very different sizes of numbers. DINO's numbers are about **8× larger** than YOLO's P5 numbers.
  - RMS (the typical size of a number) is 2.5 for DINO and 0.32 for P5.
- **Normalization** rescales numbers to a standard range: average 0, spread 1.
  - **LayerNorm (LN):** normalizes each token (each patch's 384 numbers) separately.
  - **GroupNorm (GN):** normalizes the projected 64-channel map of each image. Unlike BatchNorm, it does not depend on the batch size.
- **My midterm intuition:** normalization makes the two streams comparable, so the detector can use DINO better.
- **The 5 versions ("arms").** Each changes exactly **one** thing:

| Arm | LayerNorm | GroupNorm | Gate |
|---|:---:|:---:|:---:|
| No norm | – | – | – |
| LN only | ✓ | – | – |
| GN only | – | ✓ | – |
| **LN+GN** (my midterm design) | ✓ | ✓ | – |
| Gated (older design) | – | – | ✓ |

- A **gate** is a learnable "volume knob" for each DINO channel. The model learns how much of each channel to let through.
- This kind of study is called an **ablation**: change one part at a time and see what happens.

> **Say it:** "Each arm changes exactly one factor, so any difference in accuracy comes from that factor."

### 1.6 Two fixes that make the comparison fair
- **Fix 1, a bug fix.** The old midterm code copied the pretrained weights to the **wrong channel positions**, so YOLO's neck was reading the wrong channels. Now P5's channels come first, DINO's are appended, and every weight is copied to its exact position.
- **Fix 2, zero initialization.** The new weights that read the DINO channels **start at zero**. At the start of training, every fused model gives *exactly* the same output as plain YOLOv8n. So any difference after training comes from learning, not from a damaged start. A unit test checks this (the difference is exactly 0).

> **Say it:** "At step zero every fused model is exactly the baseline, and a unit test checks this."

---

## Part 2: How I measure "better"

### 2.1 The data
- **PASCAL VOC:** 20 everyday classes (person, car, dog, cat, chair, bird…).

| Split | Images | Used for |
|---|---|---|
| Train | 14,041 | learning |
| Validation ("dev") | 2,510 | choosing the best epoch (*checkpoint*) |
| Test | 4,952 | the final score, used **once** per run, so there is no cheating |

- **VisDrone:** drone photos with very small objects. This is the "far domain" test.
- **10% subset:** 1,404 training images, to test what happens with few labels.

> **Say it:** "I pick the checkpoint on the validation split and score each model once on the test split."

### 2.2 The metric: mAP50-95
- **IoU (Intersection over Union)** is how much the predicted box overlaps the true box. 1 is perfect; 0 is no overlap.
- A detection counts as **correct** if the class is right **and** the IoU is above a threshold.
- **Precision:** of all my detections, how many are correct.
- **Recall:** of all real objects, how many I found.
- **AP (Average Precision):** one number per class that combines precision and recall.
- **mAP:** the mean of AP over the 20 classes.
- **mAP50** uses the IoU threshold 0.5, which is *loose*.
- **mAP50-95** averages over the thresholds 0.50, 0.55, …, 0.95. It is *strict*, because it also rewards precise boxes. **This is my main metric.**
- A **"point"** is one percentage point. "+10.5 points" means 57.5% → 68.0%.

> **Say it:** "mAP50-95 is detection accuracy averaged over loose and strict box-overlap thresholds; higher is better."

### 2.3 Seeds: training is random
- Training the same model twice gives slightly different results, because the starting values and the data order are random. Each repeat is called a **seed**.
- I trained every version **3 times** and report **mean ± standard deviation**. For example, the baseline is **57.5 ± 0.4**.

> **Say it:** "Every number is the mean of three training runs, so I know how big the random noise is."

### 2.4 Confidence intervals: "is the difference real?"
- The test set is only a *sample* of possible images. Different images would give slightly different numbers.
- **Bootstrap:**
  1. Build 400 "new" test sets by randomly drawing 4,952 images *with replacement*.
  2. On each one, compute the difference between the two models.
  3. The middle 95% of those 400 differences is the **95% confidence interval (CI)**.
- **Paired** means both models are always compared on the *same* resampled images.
- If the CI does **not include 0**, the difference is real. For example, **+10.5 [9.8, 11.2]** is clearly real.

> **Say it:** "The 95% confidence interval comes from resampling test images. If it excludes zero, the difference is real."

### 2.5 Equivalence: "are they really the same?"
- "Not significant" does **not** prove "the same". It could mean "not enough data".
- To show two versions are *practically the same*, I check that the **90% CI lies inside a small margin of ±0.5 points**. That margin is about the size of the seed noise. This is called an **equivalence test** (TOST).
- **Result:** LN, GN and LN+GN versus no norm all lie inside **±0.33**, so they are **equivalent**.
- **Holm correction:** when you run several tests, you adjust the p-values so that one lucky result doesn't fool you. The gated arm still passes.

> **Say it:** "The normalization arms are statistically equivalent to no normalization, within ±0.5 points."

---

## Part 3: The experiments, in simple words

| Experiment | Simple question | Answer |
|---|---|---|
| **Main:** 6 versions × 3 seeds | Does fusion help? Does normalization matter? | Fusion **+10.5**. Normalization: **no** |
| **Mechanism logging** | How much does the detector *listen* to DINO? | 32–60% of the first neck layer's input, depending on the normalization, yet **the same accuracy** |
| **Global control** | Does it need *where* things are, or only *what* the image contains? | **Where**: two-thirds of the gain needs the spatial map |
| **Gate control** | Why is the gated version slightly better? | Both the gate itself (+0.32) and its larger starting values (+0.41) |
| **YOLOv8s** (3.5× bigger CNN) | Is the gain just "a bigger model"? | **No.** Even a cheap fusion version beats YOLOv8s |
| **Resolution sweep** | How small can DINO's input be? | At 224 px: two-thirds of the gain for one-fifth of the compute |
| **10% labels** | Does it help when labels are few? | **Even more: +15.0** |
| **VisDrone** | Does it help on drone images with tiny objects? | **Only a little: +0.8** |
| **Error analysis** | What *kind* of mistakes does it fix? | Wrong labels (−74%) much more than box position (−31%) |

> **Say it:** "Every claim has its own controlled experiment: 45 runs in total."

---

## Part 4: The results, with the "why"

1. **Fusion helps a lot: +10.5 points (57.5 → 68.0).**
   *Why:* DINOv2 learned from 142 million images, so its features describe objects very well.

2. **Normalization does not matter.**
   *Why:* the DINO weights start at zero and *grow during training*, so the network finds the right scale by itself. Normalization only changes the **starting point**. The mechanism plot shows this: different arms listen to DINO differently (32–60%), yet end at the same accuracy.

3. **The spatial map matters.**
   If I average DINO's map into one vector (the "global" version), I get only **+3.5**. The full map gives **+10.5**. DINO tells YOLO not just "there is a dog in this image", but "the dog is *here*".

4. **It mostly fixes wrong labels.**
   Wrong-class errors fall by **74%**; box-position errors only by **31%**.
   *Why:* DINO's features are about *meaning*, and they are coarse (14-pixel patches).

5. **It is worth the cost.**
   - At full resolution, fusion needs 13× the compute of YOLOv8n.
   - The fair question is: why not just use a bigger YOLO?
   - YOLOv8s (bigger CNN) scores **61.8**. Fusion at 224 px scores **64.6** while using **fewer** FLOPs (20 vs 28).
   - **FLOPs** = the number of calculations, i.e. a measure of compute.

6. **Few labels → bigger help (+15.0).**
   *Why:* DINO brings knowledge learned *without* labels. With only 10% of the labels, fusion (57.2) almost matches YOLOv8n trained on *all* labels (57.5).

7. **Tiny drone objects → small help (+0.8).**
   *Why:* DINO's patches are 14 px, the context enters at the coarse P5 level, and DINO learned from ordinary photos, not drone views.

8. **The honest limitation.**
   All runs used **15 epochs** (an *epoch* is one pass over the training data). The gap between the models is still shrinking at epoch 15 (19.8 → 10.7 points). So part of the gain may be "learning faster", and I can't claim the full gap would remain with much longer training.

> **Say it:** "What matters is the spatial DINOv2 context. Normalization only changes the starting point. A learnable gate adds a little."

---

## Part 5: Words you may hear

| Word | Simple meaning |
|---|---|
| Feature map / channel | A grid of numbers the network computes; each channel is one "layer" of that grid |
| Token / patch | A small square of the image (14×14 px) and its vector of numbers |
| Self-supervised | Learning from images without human labels |
| Frozen / fine-tune | Weights not changed / weights adjusted on new data |
| Epoch / checkpoint | One pass over the training data / the saved model at some epoch |
| Ablation | Changing or removing one part to see its effect |
| Control experiment | An extra run that rules out an alternative explanation |
| Baseline | The plain model you compare against (YOLOv8n) |
| RMS | The typical size of the numbers (root mean square) |
| FLOPs / latency | Number of calculations / time per image |
| p-value | How surprising the result would be if there were no real difference |
| TIDE | A method that splits detection errors into types: wrong class, wrong box, background, missed, duplicate |
| BatchNorm vs GroupNorm | BatchNorm uses statistics of the whole batch; GroupNorm works per image |

---

## Part 6: Check yourself (answer aloud, then check)

1. **What does my model add to YOLOv8n?** DINOv2's patch features, projected to 64 channels and concatenated to P5.
2. **Why is DINOv2 frozen?** It keeps the comparison clean and cheap; only the adapter and YOLO learn.
3. **What is mAP50-95?** Detection accuracy averaged over IoU thresholds from 0.50 to 0.95.
4. **Why 3 seeds?** Training is random. Three runs show how large the noise is.
5. **What does the 95% CI [9.8, 11.2] mean?** The true gain is very likely between 9.8 and 11.2 points, so it is clearly above zero.
6. **Why say "equivalent" and not "no difference"?** "Not significant" doesn't prove "same". The 90% CIs lie inside ±0.5, which does.
7. **Why does normalization not matter?** The DINO weights start at zero and learn the right scale; normalization only changes the start.
8. **What does the global control show?** Averaging DINO to one vector gives only +3.5, so the spatial layout carries most of the gain.
9. **Why not just use YOLOv8s?** Fusion at 224 px uses fewer FLOPs than YOLOv8s and is still 3.0 points better.
10. **What is your main limitation?** Only 15 epochs; the gap is still shrinking, so part of the gain may be faster learning.

If you can answer all ten in your own words, you are ready for the speaker notes.
