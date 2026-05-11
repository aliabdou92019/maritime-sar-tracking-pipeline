# SeaDronesSee — Team Implementation Plan

## System Overview

Two complete pipelines for maritime SAR:
- **Heavy Pipeline** (Persons 1–3): Maximum accuracy, runs on ground station server
- **Light Pipeline** (Persons 4–6): Real-time speed, runs on drone hardware

Each person owns exactly **one model** and is responsible for their own files. Everything outside a single model is a **Team Task**. The SAHI dataset is already prepared — sliced 640×640 images are in `data/sahi/train/` and `data/sahi/val/`.

---

## Models at a Glance

| Person | Model | Pipeline | Role |
|---|---|---|---|
| 1 | YOLOv8x | Heavy | Object Detector |
| 2 | Swin-B | Heavy | Image Classifier |
| 3 | Bi-LSTM + SORT | Heavy | Sequence Tracker |
| 4 | YOLOv10n | Light | Object Detector |
| 5 | MobileViT-S | Light | Image Classifier |
| 6 | GRU + SORT | Light | Sequence Tracker |

---

## Team Tasks (Shared — Not Owned by Any One Person)

### Build MOT Sequences
Run `shared/build_sequences.py` once. Reads the MOT annotation JSON files and produces `data/sequences_train.npy` and `data/sequences_val.npy`. These are the training data for Persons 3 and 6. Neither tracker person needs the raw MOT video frames — only these two files.

### Class Mapping
Create `shared/class_mapping.py` with the agreed class ID dictionary used by all 6 models consistently.

### Classifier Integration Logic
Both pipelines follow the same rule for when to call the classifier, defined here so the Integration Lead can implement it consistently in `inference.py`. For every detection where YOLO's confidence score is below 0.5, crop the detected region from the original image, resize it to 224×224, and pass it to the classifier. Use the classifier's output as the final class label. For every detection where YOLO's confidence score is 0.5 or above, trust YOLO's class label directly without calling the classifier. This check happens every frame for every active detection, before the result is passed to the tracker.

### Pipeline & Demo
After all 6 models are trained, one person is assigned as the **Integration Lead** and owns these four files. The recommended choice is whoever finishes their model earliest. Do not leave it without a single owner.

- `heavy_model/inference.py` — runs the full heavy pipeline end to end
- `heavy_model/demo.py` — runs on a video and displays tracked boxes live
- `light_model/inference.py` — runs the full light pipeline end to end
- `light_model/demo.py` — same for the light pipeline

---

## Person 1 — YOLOv8x (Heavy Detector)

**Receives:** SAHI dataset at `data/sahi/` and `data/SeaDronesSee_SAHI.yaml`

**Note:** Person 1 does not generate crops. Person 2 handles their own data preparation independently.

### File 1: `heavy_model/detector/train_detector.py`
Fine-tunes YOLOv8x on the SAHI dataset. Loads `yolov8x.pt` pretrained weights and trains for 50 epochs on `data/sahi/train/images/` using `data/SeaDronesSee_SAHI.yaml` as the dataset config. Saves the best weights to `heavy_model/detector/best.pt`.

### File 2: `heavy_model/detector/summary_detector.py`
Three things:
1. Print the model architecture and total parameter count
2. Evaluate `best.pt` on the test set and report mAP50 and mAP50-95
3. A function that accepts a single image path, runs detection on it, and displays the result with bounding boxes and class labels drawn on the image

---

## Person 2 — Swin-B (Heavy Classifier)

**Receives:** Raw training images at `data/train/images/` and `data/annotations/instances_train.json`

### File 1: `heavy_model/classifier/export_gt_crops.py`
Reads the ground truth bounding boxes directly from `instances_train.json`. For each annotation, crops that region from the corresponding training image and resizes it to exactly 224×224 regardless of the original crop size — whether the box is 20×30 pixels (upscale) or 800×600 pixels (downscale), the output is always 224×224. Saves each crop to `data/gt_crops/` with the class ID encoded in the filename. Applies ±10% random box jitter when cropping so the classifier learns to handle the slight boundary imprecision of real YOLO detections at inference. This file is run once before training begins.

Person 5 uses the same `data/gt_crops/` folder — there is no reason to have separate folders since both classifiers train on the same annotations.

### File 2: `heavy_model/classifier/train_classifier.py`
Fine-tunes a Swin-B model pretrained on ImageNet on the crop dataset generated in File 1. The class label for each crop is parsed from its filename. Splits the dataset into train and validation sets internally (e.g., 85/15 split). Saves the best weights to `heavy_model/classifier/best.pth`.

### File 3: `heavy_model/classifier/summary_classifier.py`
Three things:
1. Print the model architecture and total parameter count
2. Evaluate `best.pth` on the held-out validation split and report top-1 accuracy per class
3. A function that accepts a single image path, runs classification, and returns the predicted class name and confidence score

---

## Person 3 — Bi-LSTM + SORT (Heavy Tracker)

**Receives:** `data/sequences_train.npy` and `data/sequences_val.npy` from the Team Task

**Delivers to Person 6 on Day 1:** A stub `shared/sort_tracker.py` with the interface defined but placeholder logic, so Person 6 can work in parallel without waiting for the full implementation.

### File 1: `heavy_model/tracker/train_tracker.py`
Trains a Bidirectional LSTM to predict the next frame's bounding box given a 10-frame history of `[x, y, w, h]` positions. Saves best weights to `heavy_model/tracker/best_bilstm.pth`.

The Bi-LSTM has 2 layers and a hidden size of 512. Input is 4 values per frame, output is 4 values representing the predicted next-frame box.

**Sequence Augmentation (apply to input during training only):** The MOT sequences are clean ground truth, but the tracker will receive noisy YOLO boxes at inference. Add Gaussian noise (std = 0.02 in normalised coordinates) to every `[x, y, w, h]` value, and randomly drop 10–15% of frames replacing each dropped frame with a repeat of the previous one. The target box being predicted always stays as clean ground truth. Person 6 must use these exact same parameters.

### File 2: `heavy_model/tracker/motion_model.py`
Wraps the trained Bi-LSTM into a class with a `predict(history)` method. The `predict` method accepts a list of up to 10 `[x, y, w, h]` boxes (the track's recent history) and returns the predicted next-frame `[x, y, w, h]`. This is the interface that `shared/sort_tracker.py` calls.

### File 3: `shared/sort_tracker.py` — Person 3 writes this, Person 6 reuses it
The shared SORT tracking loop. Contains the IoU cost matrix calculation, Hungarian algorithm matching, track birth and death logic, and the main tracker class. The tracker class accepts any motion model object that has a `predict(history)` method and calls it each frame — it does not hardcode Bi-LSTM or GRU inside it. Person 3 first publishes a stub version on Day 1, then replaces it with the full implementation.

### File 4: `heavy_model/tracker/summary_tracker.py`
Three things:
1. Print the model parameter count
2. Evaluate on `sequences_val.npy` and report average IoU between predicted and actual next-frame boxes
3. A visual test that plots the predicted trajectory versus the ground truth trajectory on a sample sequence

---

## Person 4 — YOLOv10n (Light Detector)

**Receives:** SAHI dataset at `data/sahi/` and `data/SeaDronesSee_SAHI.yaml`

**Note:** Person 4 does not generate crops. Person 5 handles their own data preparation independently.

### File 1: `light_model/detector/train_detector.py`
Fine-tunes YOLOv10n on the SAHI dataset. Loads `yolov10n.pt` pretrained weights and trains for 50 epochs on `data/sahi/train/images/` using `data/SeaDronesSee_SAHI.yaml` as the dataset config. Uses a larger batch size than Person 1 since the model is smaller and more fits in GPU memory. Saves the best weights to `light_model/detector/best.pt`.

### File 2: `light_model/detector/summary_detector.py`
Three things:
1. Print the model architecture and total parameter count
2. Evaluate `best.pt` on the test set and report mAP50 and mAP50-95
3. A function that accepts a single image path, runs detection on it, and displays the result with bounding boxes and class labels drawn on the image

---

## Person 5 — MobileViT-S (Light Classifier)

**Receives:** `data/gt_crops/` generated by Person 2's `export_gt_crops.py`

Person 5 uses the same crop folder as Person 2. Person 2 must run `export_gt_crops.py` before Person 5 can start training — this is the only dependency between the two.

### File 1: `light_model/classifier/train_classifier.py`
Fine-tunes a MobileViT-S model pretrained on ImageNet on the same crop dataset at `data/gt_crops/`. The class label for each crop is parsed from its filename. Splits the dataset into train and validation sets internally (e.g., 85/15 split). Saves the best weights to `light_model/classifier/best.pth`.

### File 2: `light_model/classifier/summary_classifier.py`
Three things:
1. Print the model architecture and total parameter count
2. Evaluate `best.pth` on the held-out validation split and report top-1 accuracy per class
3. A function that accepts a single image path, runs classification, and returns the predicted class name and confidence score

---

## Person 6 — GRU + SORT (Light Tracker)

**Receives:** `data/sequences_train.npy` and `data/sequences_val.npy` from the Team Task, and the stub `shared/sort_tracker.py` from Person 3 on Day 1

Person 6 builds and tests their GRU and `motion_model.py` against the stub. When Person 3 delivers the full `sort_tracker.py`, Person 6 swaps it in with no changes needed on their side.

### File 1: `light_model/tracker/train_tracker.py`
Trains a GRU to predict the next frame's bounding box given a 10-frame history of `[x, y, w, h]` positions. Saves best weights to `light_model/tracker/best_gru.pth`.

The GRU has 2 layers and a hidden size of 128. Input is 4 values per frame, output is 4 values representing the predicted next-frame box.

**Sequence Augmentation (apply to input during training only):** Add Gaussian noise (std = 0.02 in normalised coordinates) to every `[x, y, w, h]` value, and randomly drop 10–15% of frames replacing each dropped frame with a repeat of the previous one. The target box being predicted always stays as clean ground truth. These parameters must match Person 3's exactly so results are comparable.

### File 2: `light_model/tracker/motion_model.py`
Wraps the trained GRU into a class with a `predict(history)` method. The `predict` method accepts a list of up to 10 `[x, y, w, h]` boxes (the track's recent history) and returns the predicted next-frame `[x, y, w, h]`. This is the same interface that Person 3's `motion_model.py` implements, allowing both to plug into the same `shared/sort_tracker.py`.

### File 3: `light_model/tracker/summary_tracker.py`
Three things:
1. Print the model parameter count
2. Evaluate on `sequences_val.npy` and report average IoU between predicted and actual next-frame boxes
3. A visual test that plots the predicted trajectory versus the ground truth trajectory on a sample sequence

---

## Dependency Chain

```
Team Tasks (run before anyone trains):
  build_sequences.py  → sequences_train.npy   → Person 3 & 6
  class_mapping.py    → shared by all

Person 2 → export_gt_crops.py → data/gt_crops/ → Person 5
Person 3 → stub sort_tracker.py on Day 1       → Person 6 can start immediately

Person 1, 3, 4, 6 train in full parallel after team tasks are done
Person 2 runs export_gt_crops.py first (fast), then trains
Person 5 waits for Person 2's crops, then trains

All 6 done → Integration Lead assembles inference.py and demo.py
```

---

## Full File Structure

```
Seedrone/
│
├── data/
│   ├── sahi/                          ← ready (SAHI done)
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── val/images/
│   │   └── val/labels/
│   ├── gt_crops/                      ← Person 2 generates → Person 2 & 5 use
│   ├── sequences_train.npy            ← Team Task → Person 3 & 6
│   ├── sequences_val.npy
│   └── SeaDronesSee_SAHI.yaml
│
├── shared/
│   ├── build_sequences.py             ← Team Task
│   ├── sort_tracker.py                ← Person 3 writes, Person 6 uses
│   └── class_mapping.py               ← Team Task
│
├── heavy_model/
│   ├── detector/
│   │   ├── train_detector.py          ← Person 1
│   │   └── summary_detector.py        ← Person 1
│   ├── classifier/
│   │   ├── export_gt_crops.py         ← Person 2 (run first, output shared with Person 5)
│   │   ├── train_classifier.py        ← Person 2
│   │   └── summary_classifier.py      ← Person 2
│   ├── tracker/
│   │   ├── train_tracker.py           ← Person 3
│   │   ├── motion_model.py            ← Person 3
│   │   └── summary_tracker.py         ← Person 3
│   ├── inference.py                   ← Integration Lead
│   └── demo.py                        ← Integration Lead
│
└── light_model/
    ├── detector/
    │   ├── train_detector.py          ← Person 4
    │   └── summary_detector.py        ← Person 4
    ├── classifier/
    │   ├── train_classifier.py        ← Person 5
    │   └── summary_classifier.py      ← Person 5
    ├── tracker/
    │   ├── train_tracker.py           ← Person 6
    │   ├── motion_model.py            ← Person 6
    │   └── summary_tracker.py         ← Person 6
    ├── inference.py                   ← Integration Lead
    └── demo.py                        ← Integration Lead
```

---

## Execution Order

| Step | Who | What |
|---|---|---|
| 1 | Team | Run `build_sequences.py`, create `class_mapping.py` |
| 2 | Person 3 | Publish stub `shared/sort_tracker.py` interface |
| 3 | Person 2 | Run `export_gt_crops.py` to generate `data/gt_crops/` |
| 4 | Person 1, 3, 4, 6 | Train models in parallel |
| 4 | Person 2 | Train Swin-B (after step 3) |
| 5 | Person 5 | Train MobileViT-S (after step 3) |
| 6 | Person 3 | Replace stub with real `sort_tracker.py` |
| 7 | Person 3 & 6 | Finalise `motion_model.py` and plug into real SORT |
| 8 | Integration Lead | Assemble `inference.py` and `demo.py` for both pipelines |
