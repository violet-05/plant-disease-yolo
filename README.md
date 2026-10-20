# plant-disease-yolo

**English** | [简体中文](README.zh-CN.md)

Crop leaf disease classification with YOLO — a complete pipeline from cleaning public datasets
through training and evaluation to a live webcam demo.

**17 classes**: 7 apple diseases + 4 corn + 4 citrus + rice blast + a catch-all "other crop leaf".

---

## What this project is actually about

The starting point was concrete: point a camera at a leaf labelled "leaf spot" and the model
confidently answers "apple scab".

The root cause turned out not to be a weak model, but that **an object detector has no way to say
"I don't know this"** — it must pick the most similar of the N classes it was given.
Three controlled experiments:

| What was tested | Result | Conclusion |
|---|---|---|
| Image quality lost to re-photographing a screen | only 7.6% flipped | not the cause |
| Recognising the 6 known classes | top-1 91.5% | actually fine |
| **Meeting an unseen disease** | **78% / 89% answered "apple scab"** | **the real problem** |

So the project does two things:

1. **Switches the task from detection to whole-leaf classification** — easier to learn, and a
   better match for the "photograph a leaf → what's wrong with it" use case.
2. **Adds explicit reject classes** (`其他苹果病害` / `其他作物叶片`) — letting the model say
   "this is an apple disease, but not one of the ones I know" instead of guessing.

The outcome:

| | old detection model | this project's classifier |
|---|---|---|
| Unseen disease from outside the training set | **32% hard-answer "apple scab"** | **81–88% correctly rejected** |
| Can it say "this isn't an apple leaf"? | no | yes |
| Can it say "this leaf is healthy"? | no | yes |

---

## Quickstart

```bash
pip install ultralytics==8.4.173    # pin this version — other versions can fail to load the weights
pip install -r requirements.txt
```

### 1. Get the data

Datasets are not committed (large, and each carries its own third-party licence).
See `tools/download_roboflow.py` and `tools/harvest_apple.py` for the sources, and put them under
`datasets/raw/`.

### 2. Inspect image quality

**Don't skip this.** Public datasets are far worse than you'd expect (see *Dataset quality traps*
below). Run it first:

```bash
python tools/qa_image_quality.py datasets/raw/<some-source>/
```

### 3. Build the dataset

```bash
python tools/build_apple_cls.py
```

Writes `datasets/apple_cls/{train,val,holdout}/` and packs a zip for uploading to a cloud box.

### 4. Train

```bash
python tools/train_apple_cls.py --size m                 # local
bash tools/cloud_train_resume.sh m <run-name> <data-dir> # cloud, with resume
```

### 5. Evaluate

```bash
python tools/val_apple_cls.py models/<your-model>.pt      # per-class recall + Chinese confusion matrix
python tools/probe_cls_unknown.py models/<your-model>.pt  # rejection-rate acceptance test
```

### 6. Webcam demo

```bash
python tools/webcam_dual_demo.py
```

Two tracks at once: the detector boxes the lesion, the classifier names the disease, and low
confidence shows as "uncertain".

---

## Configuration

All paths are resolved by `tools/config.py` — there are no hardcoded absolute paths anywhere in the
repo. Run it to see what it resolved:

```bash
python tools/config.py
```

| Environment variable | Purpose |
|---|---|
| `PLANT_DISEASE_ROOT` | Point at a project root elsewhere (e.g. datasets on another drive) |
| `PLANT_DISEASE_FONT` | Explicit CJK font file, if auto-detection picks the wrong one |
| `ROBOFLOW_API_KEY` | Roboflow key; alternatively put it in `.secrets/roboflow_key.txt` |

---

## Repository layout

```
tools/
├── config.py                  # path / font / credential resolution — the single source
├── search_roboflow_apple.py   # search Roboflow Universe (auto-discriminates fruit vs leaf)
├── harvest_apple.py           # batch download (picks export format by project type)
├── qa_image_quality.py        # image QA: broken images / product shots / fruit
├── build_apple_cls.py         # dataset build: de-augment / anti-leak / balance / QA / pack
├── class_names.py             # class map (order == index, single source of truth)
├── train_apple_cls.py         # training
├── val_apple_cls.py           # per-class recall + Chinese confusion matrix + threshold calibration
├── probe_cls_unknown.py       # rejection-rate acceptance test for unseen diseases
├── webcam_dual_demo.py        # two-track webcam demo
├── cloud_train_resume.sh      # cloud training (env self-check + resume)
└── diag_*.py                  # diagnostic scripts (every conclusion here came out of these)
```

---

## Dataset quality traps

**This is the most useful part of the project.** Public plant-disease datasets are full of traps.
We hit nearly all of them.

### 1. The advertised image count is fiction

`apple_catherine` claims 26,412 images; after stripping `.rf.<hash>` augmented duplicates only
**1,586 unique originals** remain. Every judgement based on image count is wrong.

### 2. The official split leaks

Augmented copies of the same original land on both the train and valid sides (measured
train ∩ valid = 351 originals). **Reusing the official split inflates val by 10–20 points.**
You must regroup by original.

We measured the cost: on the same data, splitting by source gives top-1 = **51.4%**, splitting by
random image gives **98.8%** — the latter is fake.

### 3. Augmentation can be destructive

In `apple_catherine`, the `Apple Scab` / `Black Rot` / `Cedar Apple Rust` classes were given a
**pixelation augmentation**, invalidating the whole class (after tightening the threshold, 250
originals left exactly 1 usable image).

The test: downscale to a ≤128 grid with NEAREST and count unique colours (normalise size first, or
small images get falsely rejected). Real photographs measure ≥5,000 colours; spoiled ones ≤1,500.

### 4. "Apple disease" datasets are mostly photographing fruit

- Half of `apple_catherine`'s 10 classes are **fruit and stem diseases**, not leaf.
- `apple-1` has a class called `Marssonia blotch` — nearly the same name as our brown spot —
  but **it is entirely fruit**.
- 45–58% of `citrus_research` is white-background oranges.

**You have to look at the images; class names are not enough.** After every build we montage 8–9
samples per class and look at them — the only way to catch a mis-merged class or fruit mixed into
a leaf class.

One criterion that *can* be automated: the green-pixel ratio. In the citrus set the "healthy" class
has a median of 0.07 (fruit) while "greening" has 0.46 (leaf); a threshold of 0.12 removes most
fruit.

### 5. Scraped datasets are noisy

60% of PlantDoc's `Apple leaf` class is watermarked white-background product shots. Its disease
classes (Apple Scab Leaf / Apple rust leaf / the Corn series) are usable, though.

### 6. Lesion crops are not whole leaves

`plantseg`'s lesion-box crops have a median size of just 43px and a composition completely unlike a
whole leaf, so they're unusable for classification (79% of crops are under 96px).

---

## Known weaknesses

Recorded honestly, not glossed over:

- **`苹果黑星病` (apple scab) is the weakest class** (in-domain recall 54.5%) — only 98 training
  images, the thinnest class in the table.
- **More classes, weaker rejection**: with 8 classes the correct-rejection rate on unseen diseases
  is 88%; expanded to 17 it drops to 81%. More options make "just guess one" more tempting.
  (Measured with `tools/probe_cls_unknown.py` on 26 apple-mosaic-virus holdout images.)
- **The in-domain val number is not trustworthy**: the 0.936 figure in this repo contains split
  leakage. **The honest number is 51.4% cross-source, from the 8-class version.**
- **About 20% fruit has leaked into the citrus classes** (the green filter only catches part of it).
- The demo scripts need a camera, so they only run locally.

---

## Environment

- ultralytics **8.4.173** (must match the training version, or `best.pt` may fail to load)
- PyTorch (GPU build)
- The local demo runs on a GTX 1650 4GB (detection imgsz=640 + classification imgsz=384, ~16 FPS)

## Licence

Code is [MIT licensed](LICENSE).

**Datasets are not in this repository**; each source has its own licence (mostly CC BY 4.0) — follow
the original dataset's terms.

Model weights live in the companion repo:
[plant-disease-yolo-models](https://github.com/violet-05/plant-disease-yolo-models)