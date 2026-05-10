# SeaDronesSee Dual Multi-Model Pipeline: Team Implementation Plan

## Project Overview

A team of 6 builds **two complete multi-model tracking systems** for maritime drone SAR:

| System | Philosophy | Target |
|---|---|---|
| **Heavy Model** | Maximum accuracy, no resource limit | Server / Ground Station |
| **Lightweight Model** | Minimum latency, edge-optimized | On-drone inference |

Each person owns one model independently. The two pipelines run as **fully separate standalone systems** — no fusion between them.

---

## Team Assignments

| Person | Model | System |
|---|---|---|
| **1** | Object Detector (Heavy) | Heavy |
| **2** | Image Classifier (Heavy) | Heavy |
| **3** | Sequence Tracker (Heavy) | Heavy |
| **4** | Object Detector (Light) | Lightweight |
| **5** | Image Classifier (Light) | Lightweight |
| **6** | Sequence Tracker (Light) | Lightweight |

---

## Model Candidates & Recommendations

### Slot 1 — Heavy Object Detector

| Model | mAP (COCO) | Strength |
|---|---|---|
| **YOLOv8x** ✅ Recommended | 53.9 | Best ecosystem, mature TF export, versatile |
| RT-DETR-X | ~54.8 | Transformer-based detector, no NMS |

**Recommendation: YOLOv8x** — best overall ecosystem with officially supported TF SavedModel/TFLite export via Ultralytics. The most mature framework-agnostic option with extensive documentation for both PyTorch and TF workflows.

---

### Slot 2 — Heavy Image Classifier (ViT)

| Model | ImageNet Top-1 | Params | Strength |
|---|---|---|---|
| **Swin-B** ✅ Recommended | 83.5% | 88M | Hierarchical, linear complexity, best for high-res drone images |
| DeiT-Base | 81.8% | 86M | Data-efficient, well-studied |
| ViT-B/16 | 81.8% | 86M | Classic, stable, GPU-optimized |
| EfficientNet-B7 (CNN baseline) | 84.3% | 66M | Strong non-transformer option |

**Recommendation: Swin-B** — its hierarchical shifted-window attention is specifically superior for high-resolution imagery (your frames are 3840×2160), and it scales linearly with image size instead of quadratically.

```python
# PyTorch (timm)
import timm
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=5)

# TensorFlow (pip install tfswin)
from tfswin import SwinTransformerBase
model = SwinTransformerBase(include_top=False)
```

---

### Slot 3 — Heavy Sequence Tracker

| Model | Strength |
|---|---|
| **Bi-LSTM (2 layers, hidden=512)** ✅ Recommended | Reads both past and future context in sequences |
| Standard LSTM (2 layers, hidden=256) | Simpler, slightly less accurate |
| Transformer Tracker (small) | Attention-based motion modeling |

**Recommendation: Bidirectional LSTM** — since we train on full sequences (not live), a Bi-LSTM can look at both past and future frames during training, learning richer motion patterns. At inference it falls back to unidirectional mode using only the past T=10 frames.

```python
# PyTorch
nn.LSTM(input_size=4, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)

# TensorFlow/Keras
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=False))
```

---

### Slot 4 — Lightweight Object Detector

| Model | mAP (COCO) | Params | Latency |
|---|---|---|---|
| **YOLOv10n** ✅ Recommended | 38.5 | 2.3M | ~1.84ms |
| YOLOv8n | 37.3 | 3.2M | ~1.47ms |
| YOLOv5n | 28.0 | 1.9M | Very fast |

**Recommendation: YOLOv10n** — smallest parameter count with competitive accuracy, NMS-free design lowers post-processing overhead on the drone's embedded processor, and officially supports TF SavedModel/TFLite export via Ultralytics.

---

### Slot 5 — Lightweight Image Classifier

| Model | ImageNet Top-1 | Params | Device Throughput |
|---|---|---|---|
| **MobileViT-S** ✅ Recommended | 78.4% | 5.6M | Good hybrid CNN+ViT, best TF support |
| MobileNetV3-Large (CNN baseline) | 75.2% | 5.4M | Extremely fast, built into tf.keras |

**Recommendation: MobileViT-S** — best lightweight option with official TF support via Apple's `ml-cvnets` and `keras_cv`. Its hybrid CNN+ViT design captures both local textures and global context, ideal for classifying small objects like swimmers from drone crops.

```python
# PyTorch (timm)
import timm
model = timm.create_model('mobilevit_s', pretrained=True, num_classes=5)

# TensorFlow (keras_cv)
import keras_cv
model = keras_cv.models.MobileViT(include_rescaling=True, num_classes=5)
```

---

### Slot 6 — Lightweight Sequence Tracker

| Model | Params | Strength |
|---|---|---|
| **GRU (2 layers, hidden=128)** ✅ Recommended | ~200K | Fewer gates than LSTM = faster, fewer params |
| LSTM (1 layer, hidden=128) | ~200K | Standard baseline |
| Linear + EMA (no RNN) | ~1K | Ultra-lightweight, no training needed |

**Recommendation: GRU** — mathematically equivalent to LSTM for short sequences (T=10) but 25% fewer parameters and faster on embedded hardware due to simpler gating.

```python
# PyTorch
nn.GRU(input_size=4, hidden_size=128, num_layers=2, batch_first=True)

# TensorFlow/Keras
tf.keras.layers.GRU(128, return_sequences=False)
```

---

## Critical: Cross-Framework Interface Contract

Since team members may use different frameworks, all 6 models **MUST export to ONNX** and **MUST respect this I/O contract**:

### Detector Output Format (Persons 1 & 4)
```
Output: List of detections per frame
Each detection: [x_center_norm, y_center_norm, width_norm, height_norm, confidence, class_id]
- All coordinates normalized to [0, 1] relative to image dimensions
- class_id: 0=swimmer, 1=boat, 2=jetski, 3=life_saving_appliance, 4=buoy
- File export: JSON  →  {"image_id": int, "detections": [[x,y,w,h,conf,cls], ...]}
```

### Classifier Output Format (Persons 2 & 5)
```
Input:  Cropped image patch, resized to 224x224, normalized
Output: [class_id (int), confidence (float)]
- class_id uses the SAME 5-class mapping as the detector
- File export: JSON  →  {"crop_id": int, "class_id": int, "confidence": float}
```

### Tracker Output Format (Persons 3 & 6)
```
Input:  Sequence of T=10 normalized bounding boxes [x,y,w,h] from detector+classifier
Output: Predicted next-frame box [x,y,w,h] + track_id (int)
- File export: CSV  →  frame_id, track_id, x, y, w, h
```

> [!IMPORTANT]
> Every person must create a wrapper function `predict(input) -> output` in the standard format above, regardless of which internal framework they use. This is the only contract needed for fusion.

---

## Training Data (Shared by All 6)

| Model Slot | Training Data | Format |
|---|---|---|
| Detector (Heavy + Light) | OD v2: 8,930 train / 1,547 val images | COCO JSON → YOLO txt |
| Classifier (Heavy + Light) | OD v2 GT crops: ~40K cropped patches | ImageFolder (class subfolders) |
| Tracker (Heavy + Light) | MOT JSON annotations only (~11MB) | `.npy` sequences via `build_sequences.py` |

> [!NOTE]
> Persons 3 and 6 (trackers) do NOT need the 27.6GB MOT image dataset. They only need the annotation JSON files. Sequences are extracted by a shared `build_sequences.py` script.

---

## Shared Script: `build_sequences.py`

Both tracker team members (Persons 3 & 6) use the same script. It:
1. Loads `instances_train_objects_in_water.json` and `instances_val_objects_in_water.json`
2. Groups by `video_id` → `track_id`, sorts by `frame_index`
3. Normalizes `[x, y, w, h]` by image resolution (3840x2160)
4. Applies class mapping: `{1→0, 2→0, 3→1, 6→3}` (MOT → OD class IDs)
5. Creates sliding windows of T=10, saves `sequences_train.npy` + `sequences_val.npy`

Person 3 trains Bi-LSTM on this data. Person 6 trains GRU on the exact same data.

---

## Pipeline Architectures (Independent)

### Heavy Model Pipeline
```
[ Raw Video Frame ]
        │
        ▼
  [ YOLOv8x ]  ──►  Primary bounding boxes + classes
        │
        ▼
  [ Swin-B ]   ──►  Re-classifies low-confidence detections
        │
        ▼
  [ Bi-LSTM ]  ──►  Smooths boxes, assigns track IDs
        │
        ▼
[ Final Tracked Output: high-accuracy ]
```

### Lightweight Model Pipeline
```
[ Raw Video Frame ]
        │
        ▼
  [ YOLOv10n ]   ──►  Fast bounding boxes + classes (NMS-free)
        │
        ▼
  [ MobileViT-S ] ──►  Re-classifies low-confidence detections
        │
        ▼
  [ GRU ]         ──►  Smooths boxes, assigns track IDs
        │
        ▼
[ Final Tracked Output: drone-optimized ]
```

> [!NOTE]
> The two pipelines are **completely independent**. Heavy runs on a server or ground station for maximum accuracy. Light runs on-board the drone for real-time low-latency inference. They are evaluated separately and do not communicate with each other.

---

## File Structure (Full Team)

```
Seedrone/
├── data/
│   ├── seadronessee.yaml           # YOLO dataset config
│   ├── sequences_train.npy         # Tracker training data (shared)
│   └── sequences_val.npy           # Tracker validation data (shared)
│
├── heavy_model/
│   ├── detector/                   # Person 1 — YOLOv8x
│   │   └── train_detector.py
│   ├── classifier/                 # Person 2 — Swin-B
│   │   ├── train_classifier.py
│   │   └── model.py / best.pth
│   ├── tracker/                    # Person 3 — Bi-LSTM
│   │   ├── train_tracker.py
│   │   └── model.py / best.pth
│   ├── inference.py                # Heavy end-to-end pipeline
│   └── demo.py                     # Heavy visual demo
│
├── light_model/
│   ├── detector/                   # Person 4 — YOLOv10n
│   │   └── train_detector.py
│   ├── classifier/                 # Person 5 — MobileViT-S
│   │   ├── train_classifier.py
│   │   └── model.py / best.pth
│   ├── tracker/                    # Person 6 — GRU
│   │   ├── train_tracker.py
│   │   └── model.py / best.pth
│   ├── inference.py                # Light end-to-end pipeline
│   └── demo.py                     # Light visual demo
│
└── shared/
    ├── build_sequences.py          # Shared by Persons 3 & 6
    ├── class_mapping.py            # Shared class ID mapping dict
    └── interface.py                # Standardized predict() wrappers
```

---

## Training Time Estimates Per Person

| Person | Model | Est. Training Time | Hardware |
|---|---|---|---|
| 1 | YOLOv8x | 5–8 hrs | GPU (Colab T4 / RTX 3060) |
| 2 | Swin-B | 3–5 hrs | GPU |
| 3 | Bi-LSTM | 20–40 min | CPU or GPU |
| 4 | YOLOv10n | 2–4 hrs | GPU |
| 5 | MobileViT-S | 1–2 hrs | GPU |
| 6 | GRU | 10–20 min | CPU or GPU |

---

## Verification Plan

### Per-Model Metrics (Individual Responsibility)
| Model | Metric | Target |
|---|---|---|
| Detector Heavy | mAP50 on OD val | > 60% |
| Detector Light | mAP50 on OD val | > 45% |
| Classifier Heavy | Top-1 Accuracy on OD crops | > 75% |
| Classifier Light | Top-1 Accuracy on OD crops | > 65% |
| Tracker Heavy | Avg IoU next-frame prediction | > 0.70 |
| Tracker Light | Avg IoU next-frame prediction | > 0.65 |

### Full Pipeline Metrics (Per System, Independent)
- **Heavy pipeline:** MOTA / HOTA on MOT val clips via SeaDronesSee evaluation scripts
- **Light pipeline:** MOTA / HOTA on MOT val clips via SeaDronesSee evaluation scripts
- **Speed comparison:** FPS of Heavy `demo.py` vs. Light `demo.py` on the same video clip
- **Visual demo:** Each pipeline runs its own `demo.py` on a MOT video independently

---

## Open Questions for Team Discussion

> [!WARNING]
> **Q1 — Class Mapping Consistency:** The MOT dataset has 4 classes with different IDs than the OD dataset. Persons 3 & 6 MUST use the shared `class_mapping.py` exactly. Mismatched class IDs will silently break tracking output in both pipelines.

> [!NOTE]
> **Q2 — Crop Generation for Classifiers:** Persons 2 & 5 need cropped patches from the OD dataset. Should one person run `export_yolo_features.py` and share the crops folder with the team, or should each do it independently?
