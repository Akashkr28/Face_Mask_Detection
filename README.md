# Face Mask Detection

Real-time face mask detection with YOLOv8. Detects three classes:
- `with_mask` (green)
- `without_mask` (red) — safety-critical
- `mask_weared_incorrect` (orange)

## Quick start

```bash
pip install -r requirements.txt

# 1. Download dataset (needs ~/.kaggle/kaggle.json)
python scripts/download_dataset.py

# 2. Convert VOC XML → YOLO format
python scripts/convert_voc_to_yolo.py

# 3. Split 70/20/10
python scripts/split_dataset.py

# 4. (Optional) augment training set ×3
python scripts/augment.py --multiplier 3

# 5. Train
python train.py

# 6. Evaluate on test set
python evaluate.py

# 7a. Real-time webcam (OpenCV)
python detect_webcam.py

# 7b. Streamlit demo UI
streamlit run app.py
```

## Project structure

```
Face_Mask_Detection/
├── data/
│   ├── raw/                  # Raw Kaggle dataset
│   ├── yolo/                 # YOLO-format dataset
│   └── dataset.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── convert_voc_to_yolo.py
│   ├── split_dataset.py
│   ├── augment.py
│   ├── export_model.py       # ONNX / TFLite export
│   └── gradcam.py            # Explainability
├── train.py
├── evaluate.py
├── detect_webcam.py
├── app.py                    # Streamlit UI
└── requirements.txt
```

## Target metrics

| Metric | Target |
|---|---|
| mAP@0.5 | > 90% |
| mAP@0.5:0.95 | > 65% |
| Precision | > 88% |
| Recall | > 85% |
| FPS (inference) | > 25 |
| `without_mask` recall | > 92% |

## Webcam controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `+` / `-` | Adjust confidence threshold |

## Extensions

```bash
# Export to ONNX
python scripts/export_model.py --format onnx

# Grad-CAM explainability
pip install grad-cam
python scripts/gradcam.py --image path/to/face.jpg
```
