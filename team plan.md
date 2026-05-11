# SeaDronesSee — Team Implementation Plan

## System Overview

Two complete pipelines for maritime SAR:
- **Heavy Pipeline** (Persons 1–3): Maximum accuracy, runs on ground station server
- **Light Pipeline** (Persons 4–6): Real-time speed, runs on drone hardware

Each person owns exactly **one model** and is responsible for three files. Everything outside a single model is a **Team Task**.

> [!NOTE]
> The SAHI dataset is already prepared. The sliced 640×640 training images are ready in `data/sahi/train/` and `data/sahi/val/`. Persons 1 and 4 use this directly.

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
Run `shared/build_sequences.py` once. Reads the MOT annotation JSON files and produces `data/sequences_train.npy` and `data/sequences_val.npy`. These are the training data for Persons 3 and 6. Neither tracker person needs the raw MOT video frames — only these files.

### Class Mapping
Create `shared/class_mapping.py` with the agreed class ID dictionary used by all 6 models consistently.

### Pipeline & Demo
After all 6 models are trained, assemble:
- `heavy_model/inference.py` — runs the full heavy pipeline end to end
- `heavy_model/demo.py` — runs on a video and displays tracked boxes live
- `light_model/inference.py` — runs the full light pipeline end to end
- `light_model/demo.py` — same for the light pipeline

---

## Person 1 — YOLOv8x (Heavy Detector)

**Receives:** SAHI dataset at `data/sahi/` and `data/SeaDronesSee_SAHI.yaml`

**Delivers to Person 2:** Cropped object images at `data/yolo_crops/heavy/`

### File 1: `heavy_model/detector/train_detector.py`
Fine-tunes YOLOv8x on the SAHI dataset. Saves the best weights to `heavy_model/detector/best.pt`.

### File 2: `heavy_model/detector/summary_detector.py`
Three things:
1. Print model architecture and parameter count
2. Evaluate the trained model on the test set and report mAP50 and mAP50-95
3. A function that accepts a single image path, runs detection, and displays the result with bounding boxes drawn

### File 3: `heavy_model/detector/export_crops.py`
Runs the trained model on the full test set. For every detected bounding box, crops that region from the original image, resizes it to 224×224, and saves it to `data/yolo_crops/heavy/`. The filename encodes the class ID so Person 2 can use it directly as training data.

---

## Person 2 — Swin-B (Heavy Classifier)

**Receives:** Crops from Person 1 at `data/yolo_crops/heavy/`

### File 1: `heavy_model/classifier/train_classifier.py`
Fine-tunes a Swin-B model pretrained on ImageNet on the crop dataset produced by Person 1. Saves the best weights to `heavy_model/classifier/best.pth`.

### File 2: `heavy_model/classifier/summary_classifier.py`
Three things:
1. Print model architecture and parameter count
2. Evaluate on a held-out portion of the crop dataset and report top-1 accuracy per class
3. A function that accepts a single crop image path, runs classification, and returns the predicted class name and confidence score

---

## Person 3 — Bi-LSTM + SORT (Heavy Tracker)

**Receives:** `data/sequences_train.npy` and `data/sequences_val.npy` from the Team Task

**Delivers to Person 6:** `shared/sort_tracker.py` — the shared SORT matching logic

### File 1: `heavy_model/tracker/train_tracker.py`
Trains a Bidirectional LSTM to predict the next frame's bounding box given a 10-frame history of `[x, y, w, h]` positions. Saves best weights to `heavy_model/tracker/best_bilstm.pth`.

The Bi-LSTM has 2 layers and a hidden size of 512. Input is 4 values per frame, output is 4 values (the predicted next-frame box).

### File 2: `shared/sort_tracker.py` ← **Person 3 writes this. Person 6 reuses it.**
The shared SORT tracking loop. Contains the IoU cost matrix calculation, Hungarian algorithm matching, track birth/death logic, and the main tracker class. Accepts any motion model through a standard `predict(history)` interface — it does not hardcode Bi-LSTM or GRU inside it.

### File 3: `heavy_model/tracker/summary_tracker.py`
Three things:
1. Print model parameter count
2. Evaluate on the val sequences — report average IoU between predicted and actual next-frame boxes
3. A visual test that plots the predicted trajectory versus the ground truth trajectory on a sample sequence

---

## Person 4 — YOLOv10n (Light Detector)

**Receives:** Same SAHI dataset as Person 1

**Delivers to Person 5:** Cropped object images at `data/yolo_crops/light/`

Same three-file structure as Person 1. The only differences:
- Model is `yolov10n.pt` instead of `yolov8x.pt`
- Batch size is larger (model is smaller, more fits in GPU memory)
- Crops are saved to `data/yolo_crops/light/` instead of `heavy/`

### Files:
- `light_model/detector/train_detector.py`
- `light_model/detector/summary_detector.py`
- `light_model/detector/export_crops.py`

---

## Person 5 — MobileViT-S (Light Classifier)

**Receives:** Crops from Person 4 at `data/yolo_crops/light/`

Same two-file structure as Person 2. The only difference is the model is MobileViT-S instead of Swin-B.

### Files:
- `light_model/classifier/train_classifier.py`
- `light_model/classifier/summary_classifier.py`

---

## Person 6 — GRU + SORT (Light Tracker)

**Receives:** `data/sequences_train.npy` and `data/sequences_val.npy` from the Team Task, and `shared/sort_tracker.py` from Person 3

### File 1: `light_model/tracker/train_tracker.py`
Trains a GRU to predict the next frame's bounding box given a 10-frame history. Saves best weights to `light_model/tracker/best_gru.pth`.

The GRU has 2 layers and a hidden size of 128. Same input/output format as the Bi-LSTM so the SORT interface is identical.

### File 2: `light_model/tracker/motion_model.py`
Wraps the trained GRU into a class with a `predict(history)` method — the interface that `shared/sort_tracker.py` expects. Person 6 plugs this into Person 3's SORT tracker to complete the light tracker.

### File 3: `light_model/tracker/summary_tracker.py`
Same three things as Person 3's summary file.

---

## Dependency Chain

Nothing can start until the Team Tasks are done. Within the team tasks, SAHI is already complete.

```
Team: build_sequences.py
        │
        ├──► Person 3 (Bi-LSTM + sort_tracker.py)
        │         └──► Person 6 (GRU plugs into sort_tracker.py)
        │
        └──► Person 6 (GRU training data — same sequences)

Person 1 (YOLOv8x) → export_crops.py → Person 2 (Swin-B)

Person 4 (YOLOv10n) → export_crops.py → Person 5 (MobileViT-S)

All 6 done → Team assembles inference.py and demo.py
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
│   ├── yolo_crops/
│   │   ├── heavy/                     ← Person 1 → Person 2
│   │   └── light/                     ← Person 4 → Person 5
│   ├── sequences_train.npy            ← Team Task → Person 3 & 6
│   ├── sequences_val.npy
│   └── SeaDronesSee_SAHI.yaml
│
├── shared/
│   ├── build_sequences.py             ← Team Task
│   ├── sort_tracker.py                ← Person 3 writes, Person 6 uses
│   └── class_mapping.py              ← Team Task
│
├── heavy_model/
│   ├── detector/
│   │   ├── train_detector.py          ← Person 1
│   │   ├── summary_detector.py        ← Person 1
│   │   └── export_crops.py            ← Person 1
│   ├── classifier/
│   │   ├── train_classifier.py        ← Person 2
│   │   └── summary_classifier.py      ← Person 2
│   ├── tracker/
│   │   ├── train_tracker.py           ← Person 3
│   │   └── summary_tracker.py         ← Person 3
│   ├── inference.py                   ← Team Task
│   └── demo.py                        ← Team Task
│
└── light_model/
    ├── detector/
    │   ├── train_detector.py          ← Person 4
    │   ├── summary_detector.py        ← Person 4
    │   └── export_crops.py            ← Person 4
    ├── classifier/
    │   ├── train_classifier.py        ← Person 5
    │   └── summary_classifier.py      ← Person 5
    ├── tracker/
    │   ├── train_tracker.py           ← Person 6
    │   ├── motion_model.py            ← Person 6
    │   └── summary_tracker.py         ← Person 6
    ├── inference.py                   ← Team Task
    └── demo.py                        ← Team Task
```

---

## Execution Order

| Step | Who | What |
|---|---|---|
| 1 | Team | Run `build_sequences.py` |
| 2 | Team | Create `class_mapping.py` |
| 3 | Person 1 & 4 | Train detectors (can run in parallel) |
| 4 | Person 1 & 4 | Run `export_crops.py` |
| 5 | Person 2 & 5 | Train classifiers (after step 4) |
| 6 | Person 3 | Train Bi-LSTM, write `sort_tracker.py` |
| 7 | Person 6 | Train GRU, write `motion_model.py` (after step 6) |
| 8 | Team | Assemble `inference.py` and `demo.py` for both pipelines |
