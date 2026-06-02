"""
Image detection tab.
Accepts an uploaded image, runs inference, shows original vs annotated
side-by-side, renders detection stats, and offers a download button.
"""

import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from core.inference import draw_boxes
from ui.components import render_stats_block, section_header, upload_placeholder


def render(model: YOLO, conf: float, imgsz: int) -> None:
    section_header("Upload an Image")

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        upload_placeholder(
            icon="🖼️",
            hint="Drag & drop an image, or click <b>Browse files</b><br>"
                 "Supports JPG, PNG, WEBP · Max 200 MB",
        )
        return

    # ── Run inference ──────────────────────────────────────────────────────────
    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Running detection..."):
        results = model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)

    annotated_rgb, counts = draw_boxes(img_bgr, results, conf)

    # ── Side-by-side images ────────────────────────────────────────────────────
    section_header("Results")
    col_orig, col_det = st.columns(2, gap="medium")

    with col_orig:
        st.markdown(
            "<div style='font-size:0.8rem; color:rgba(255,255,255,0.4); "
            "margin-bottom:6px;'>ORIGINAL</div>",
            unsafe_allow_html=True,
        )
        st.image(pil_img, use_container_width=True)

    with col_det:
        st.markdown(
            "<div style='font-size:0.8rem; color:rgba(255,255,255,0.4); "
            "margin-bottom:6px;'>DETECTED</div>",
            unsafe_allow_html=True,
        )
        st.image(annotated_rgb, use_container_width=True)

    # ── Stats ──────────────────────────────────────────────────────────────────
    section_header("Statistics")
    render_stats_block(counts)

    # ── Download ───────────────────────────────────────────────────────────────
    annotated_pil = Image.fromarray(annotated_rgb)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as buf:
        annotated_pil.save(buf.name, quality=95)
        with open(buf.name, "rb") as f:
            st.download_button(
                "⬇️  Download Annotated Image",
                f,
                file_name="mask_detection_result.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
