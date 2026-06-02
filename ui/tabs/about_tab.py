"""
About tab — project overview, model specs, performance tables, use cases.
Pure presentation; no inference code.
"""

import streamlit as st

from ui.components import section_header

_USE_CASES = [
    ("🏥", "Healthcare",    "Hospital entrances, ICUs, operation theaters"),
    ("🏭", "Manufacturing", "PPE compliance on factory floors"),
    ("✈️", "Transport",     "Airport & transit hub screening"),
    ("🏢", "Corporate",     "Office & meeting room access control"),
]


def render() -> None:
    col_a, col_b = st.columns(2, gap="large")

    # ── Left column ────────────────────────────────────────────────────────────
    with col_a:
        section_header("About This Project")
        st.markdown("""
        Real-time face mask detection system built with **YOLOv8s** and deployed
        with **Streamlit**. Designed for safety-critical environments where mask
        compliance must be monitored automatically.

        The model is specifically tuned to minimise **false negatives on the
        `without_mask` class** — missing someone without a mask is more costly
        than a false alarm.
        """)

        section_header("Model Architecture")
        st.markdown("""
        - **Backbone:** CSPDarknet with C2f blocks
        - **Neck:** PANet-FPN (multi-scale feature fusion)
        - **Head:** Decoupled detection heads (P3 / P4 / P5)
        - **Parameters:** 11.1 M
        - **GFLOPs:** 28.4
        - **Input size:** 640 × 640
        """)

        section_header("Training")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Base model | yolov8s.pt (COCO pretrained) |
        | Dataset | 853 images → ~2,400 after augmentation |
        | Epochs | 55 (early stopping at epoch 35) |
        | Batch size | 16 |
        | Optimizer | AdamW (lr = 0.001) |
        | Hardware | Google Colab T4 GPU |
        | Training time | ~20 minutes |
        """)

    # ── Right column ───────────────────────────────────────────────────────────
    with col_b:
        section_header("Performance")
        st.markdown("""
        | Metric | Score | Target | Status |
        |---|---|---|---|
        | mAP@0.5 | 0.935 | > 0.90 | ✅ |
        | mAP@0.5:0.95 | 0.639 | > 0.65 | ≈ |
        | Precision | 0.876 | > 0.88 | ≈ |
        | Recall | 0.892 | > 0.85 | ✅ |
        | No-mask Recall | 0.921 | > 0.92 | ✅ |
        """)

        section_header("Per-Class Results")
        st.markdown("""
        | Class | Precision | Recall | mAP@0.5 |
        |---|---|---|---|
        | 🟢 With Mask | 0.961 | 0.866 | 0.920 |
        | 🔴 No Mask | 0.895 | 0.921 | 0.903 |
        | 🟠 Incorrect | 0.783 | 0.889 | 0.841 |
        """)

        section_header("Tech Stack")
        st.markdown("""
        | Layer | Tool |
        |---|---|
        | Model | YOLOv8s (Ultralytics) |
        | Deep Learning | PyTorch 2.x |
        | Computer Vision | OpenCV |
        | Augmentation | Albumentations |
        | Webcam | streamlit-webrtc + WebRTC |
        | Deployment | Streamlit Cloud |
        """)

    # ── Use-case cards (full width) ────────────────────────────────────────────
    section_header("Use Cases")
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, _USE_CASES):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="text-align:left; padding:16px;">
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-weight:600; color:white; margin-top:8px;
                            font-size:0.9rem;'>{title}</div>
                <div style='color:rgba(255,255,255,0.4); font-size:0.78rem;
                            margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
