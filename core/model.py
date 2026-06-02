"""
Model loading helpers.
All loaders are decorated with @st.cache_resource so the model is
instantiated only once per Streamlit server process.
"""

from pathlib import Path

import streamlit as st
from ultralytics import YOLO

from core.config import ONNX_WEIGHTS


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    """Load a YOLOv8 model from a .pt or .onnx file."""
    return YOLO(weights_path)


@st.cache_resource
def load_webcam_model() -> YOLO:
    """
    Load the model used by the background webcam inference thread.
    Prefers ONNX for faster CPU inference on cloud; falls back to .pt.
    """
    return YOLO(ONNX_WEIGHTS if Path(ONNX_WEIGHTS).exists() else
                "runs/detect/mask_detector/weights/best.pt")
