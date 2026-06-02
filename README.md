# 😷 Face Mask Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://face-mask-detection-a565g3luuckqxrn5fm3d8f.streamlit.app)
[![Model](https://img.shields.io/badge/Model-YOLOv8s-00BFFF?style=for-the-badge)](https://ultralytics.com)
[![mAP@0.5](https://img.shields.io/badge/mAP%400.5-93.5%25-brightgreen?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()

Real-time face mask detection system built with **YOLOv8s** and deployed with **Streamlit**. Detects three classes — with mask, without mask, and incorrectly worn mask — from images, videos, and live webcam streams.

---

## 🚀 Live Demo

**[Try it here → face-mask-detection.streamlit.app](https://face-mask-detection-a565g3luuckqxrn5fm3d8f.streamlit.app)**

| Image Detection | Video Detection | Live Webcam |
|---|---|---|
| Upload any image | Upload any video | Real-time browser webcam |
| Instant detection | Frame-by-frame analysis | Background inference thread |
| Download annotated result | Compliance trend chart | No video freeze |

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Flow](#-pipeline-flow)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Deployment](#-deployment)
- [Extensions](#-extensions)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project implements an end-to-end face mask detection pipeline for safety-critical environments such as hospitals, factories, and public spaces. The system detects three classes in real time:

| Class | Color | Description | Priority |
|---|---|---|---|
| `with_mask` | 🟢 Green | Face mask worn correctly | Normal |
| `without_mask` | 🔴 Red | No mask detected | **Safety-critical** |
| `mask_weared_incorrect` | 🟠 Orange | Mask worn incorrectly (below nose, on chin) | Warning |

### Safety-First Design
The model is specifically tuned to **minimise false negatives on `without_mask`** — missing a person without a mask is more costly than a false alarm. The `without_mask` recall target is set at **92%**, stricter than the overall recall target of 85%.

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                             │
│                                                                     │
│  Kaggle Dataset          VOC XML              YOLO Format           │
│  (853 images)    ──▶    Annotations   ──▶   (cx, cy, w, h)         │
│  3 classes               convert              normalized            │
│                                │                                    │
│                                ▼                                    │
│                    70% Train / 20% Val / 10% Test                   │
│                                │                                    │
│                                ▼                                    │
│                    Albumentations Augmentation                      │
│                    (×3 → ~2,400 training images)                    │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           TRAINING                                  │
│                                                                     │
│  YOLOv8s pretrained       Fine-tune on          Early stopping      │
│  (ImageNet weights) ──▶   mask dataset   ──▶   best.pt saved       │
│                            100 epochs                               │
│                            T4 GPU (~20 min)                         │
│                            W&B logging                              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          EVALUATION                                 │
│                                                                     │
│   mAP@0.5: 0.935    Precision: 0.876    Recall: 0.892              │
│   without_mask recall: 0.921  ← safety-critical target hit ✅       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT                                  │
│                                                                     │
│   best.pt ──▶ ONNX export ──▶ Streamlit app ──▶ Streamlit Cloud    │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│   │ Image Upload │  │ Video Upload │  │  Live Webcam (WebRTC) │    │
│   │  Detection   │  │  Frame-by-   │  │  Background inference │    │
│   │  + Download  │  │  frame stats │  │  thread (no freeze)   │    │
│   └──────────────┘  └──────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Andrewmvd Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) |
| Original size | 853 images |
| After augmentation | ~2,400 training images |
| Annotation format | Pascal VOC XML → converted to YOLO |
| Split | 70% train / 20% val / 10% test |
| Classes | `with_mask`, `without_mask`, `mask_weared_incorrect` |

### Augmentation pipeline (Albumentations)
- Horizontal flip
- Random brightness/contrast (±30%)
- Hue/saturation/value shifts
- Gaussian noise
- Motion blur
- Random rotation (±15°)
- Random scale (±20%)
- Random shadow
- CLAHE

---

## 🧠 Model Architecture

```
Input (640×640×3)
        │
        ▼
┌───────────────────┐
│   Backbone        │  CSPDarknet — extracts multi-scale features
│   (C2f blocks)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Neck (PAN-FPN)  │  Feature pyramid — fuses features across scales
│                   │  Handles faces at multiple distances/sizes
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼         ▼
 Head P3    Head P4   Head P5
 (small)   (medium)  (large)
    │         │         │
    └────┬────┘
         ▼
  Bounding boxes + Class probabilities
  (3 classes: with_mask / without_mask / mask_weared_incorrect)
```

**Model:** YOLOv8s (small)
- Parameters: 11.1M
- GFLOPs: 28.4
- Weights size: 22.5 MB
- Input: 640×640

---

## 📊 Performance Metrics

### Test Set Results

| Metric | Score | Target | Status |
|---|---|---|---|
| mAP@0.5 | **0.935** | > 0.90 | ✅ PASS |
| mAP@0.5:0.95 | **0.639** | > 0.65 | ≈ close |
| Precision | **0.876** | > 0.88 | ≈ close |
| Recall | **0.892** | > 0.85 | ✅ PASS |
| `without_mask` Recall | **0.921** | > 0.92 | ✅ PASS |
| Inference speed | ~260ms/frame (CPU M1) | — | — |

### Per-Class Results

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| with_mask | 0.961 | 0.866 | 0.920 |
| without_mask | 0.895 | 0.921 | 0.903 |
| mask_weared_incorrect | 0.783 | 0.889 | 0.841 |

### Training Details
- Base model: `yolov8s.pt` (pretrained on COCO)
- Epochs: 55 (early stopping at best epoch 35)
- Batch size: 16
- Optimizer: AdamW (lr=0.001)
- Hardware: Google Colab T4 GPU
- Training time: ~20 minutes

---

## 📁 Project Structure

```
Face_Mask_Detection/
│
├── data/
│   ├── raw/                        # Raw Kaggle dataset
│   │   ├── images/                 # Original images (853)
│   │   └── annotations/            # Pascal VOC XML files
│   ├── converted/                  # After VOC→YOLO conversion
│   │   ├── images/
│   │   └── labels/
│   ├── yolo/                       # Final YOLO dataset (split)
│   │   ├── images/
│   │   │   ├── train/              # 597 images
│   │   │   ├── val/                # 170 images
│   │   │   └── test/               # 86 images
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── dataset.yaml                # YOLOv8 dataset config
│
├── scripts/
│   ├── download_dataset.py         # Kaggle API download
│   ├── convert_voc_to_yolo.py      # Pascal VOC XML → YOLO labels
│   ├── split_dataset.py            # 70/20/10 train/val/test split
│   ├── augment.py                  # Albumentations offline augmentation
│   ├── export_model.py             # Export to ONNX / TFLite / CoreML
│   └── gradcam.py                  # Grad-CAM explainability
│
├── runs/
│   └── detect/
│       └── mask_detector/
│           └── weights/
│               ├── best.pt         # Best trained weights (22.5 MB)
│               └── best.onnx       # ONNX export (42.7 MB)
│
├── train.py                        # YOLOv8 training entry point
├── evaluate.py                     # Test set evaluation + charts
├── detect_webcam.py                # Real-time OpenCV webcam script
├── app.py                          # Streamlit web application
├── colab_train.ipynb               # Google Colab training notebook
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip
- Kaggle account + API token (`~/.kaggle/access_token`)

### Installation

```bash
git clone https://github.com/Akashkr28/Face_Mask_Detection.git
cd Face_Mask_Detection
pip install -r requirements.txt
```

### Run the Streamlit app (with pretrained weights)

```bash
streamlit run app.py
```

### Run real-time webcam detection (OpenCV)

```bash
python detect_webcam.py --weights runs/detect/mask_detector/weights/best.pt
```

#### Webcam controls
| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

---

## 🏋️ Training

### Option A — Local (CPU, slow)

```bash
# 1. Download dataset
python scripts/download_dataset.py

# 2. Convert annotations
python scripts/convert_voc_to_yolo.py

# 3. Split dataset
python scripts/split_dataset.py

# 4. Augment training set
python scripts/augment.py --multiplier 3

# 5. Train
python train.py --model yolov8s.pt --epochs 100 --batch 16
```

### Option B — Google Colab (T4 GPU, ~20 min) ✅ Recommended

1. Open `colab_train.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Add your Kaggle token in Cell 4
4. **Run all** — weights auto-save to Google Drive every 10 epochs

### Training configuration

```python
model.train(
    data='data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    cls=1.5,           # upweight classification loss
    mosaic=1.0,        # mosaic augmentation
    mixup=0.1,
    patience=20,       # early stopping
)
```

---

## 📈 Evaluation

```bash
# Evaluate on test set
python evaluate.py --weights runs/detect/mask_detector/weights/best.pt

# Evaluate with lower confidence (higher recall)
python evaluate.py --conf 0.25
```

Output includes:
- Per-class precision, recall, mAP@0.5, mAP@0.5:0.95
- PASS/FAIL against target metrics
- Recall bar chart saved to `runs/eval_recall.png`
- Confusion matrix saved by Ultralytics

---

## 🚀 Deployment

### Local Streamlit

```bash
streamlit run app.py
```

### Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set main file to `app.py`
4. Deploy

### Export to ONNX

```bash
python scripts/export_model.py \
    --weights runs/detect/mask_detector/weights/best.pt \
    --format onnx \
    --simplify
```

---

## 🔬 Extensions

### Grad-CAM Explainability
Visualize which image regions drive the model's predictions:

```bash
pip install grad-cam
python scripts/gradcam.py \
    --weights runs/detect/mask_detector/weights/best.pt \
    --image path/to/face.jpg \
    --class_id 1   # 0=with_mask, 1=without_mask, 2=incorrect
```

### Export to other formats

```bash
# TFLite (mobile/edge)
python scripts/export_model.py --format tflite --imgsz 320

# CoreML (Apple devices)
python scripts/export_model.py --format coreml
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Model | YOLOv8s (Ultralytics) | Single-shot object detector |
| Deep learning | PyTorch 2.x | Model backend |
| Computer vision | OpenCV | Frame capture & rendering |
| Augmentation | Albumentations | Bbox-safe transforms |
| Export | ONNX Runtime | Faster CPU inference |
| Web app | Streamlit | Demo UI |
| Webcam streaming | streamlit-webrtc + WebRTC | Browser webcam |
| Training hardware | Google Colab T4 GPU | ~20 min training |
| Deployment | Streamlit Cloud | Free public hosting |
| Dataset | Kaggle (Andrewmvd) | 853 annotated images |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Andrewmvd — Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [Albumentations](https://albumentations.ai)
