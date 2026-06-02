"""
Streamlit demo app for face mask detection.

Usage:
    streamlit run app.py
"""

import tempfile
import time
import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["With Mask", "No Mask", "Incorrect Mask"]
CLASS_COLORS_RGB = {0: (0, 200, 0), 1: (220, 0, 0), 2: (255, 165, 0)}
DEFAULT_WEIGHTS = "runs/detect/mask_detector/weights/best.pt"
HISTORY_LEN = 100  # frames to keep for compliance chart

# ── Module-level webcam inference state (persists across Streamlit reruns) ────
_wc_lock = threading.Lock()
_wc_latest_frame = {"img": None}
_wc_latest_result = {"boxes": [], "counts": {0: 0, 1: 0, 2: 0}}
_wc_thread_started = False


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)

@st.cache_resource
def load_webcam_model():
    """Load ONNX model for faster webcam inference."""
    onnx_path = "runs/detect/mask_detector/weights/best.onnx"
    pt_path = "runs/detect/mask_detector/weights/best.pt"
    path = onnx_path if Path(onnx_path).exists() else pt_path
    return YOLO(path)


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(image_bgr, results, conf_threshold):
    frame = image_bgr.copy()
    counts = {0: 0, 1: 0, 2: 0}
    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls)
        counts[cls_id] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color_rgb = CLASS_COLORS_RGB.get(cls_id, (180, 180, 180))
        color_bgr = color_rgb[::-1]
        label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), counts


def show_stats(counts, compliance_history=None):
    total = sum(counts.values())
    compliance = (counts[0] / total * 100) if total > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("With Mask", counts[0])
    col2.metric("No Mask", counts[1],
                delta=f"-{counts[1]}" if counts[1] else None,
                delta_color="inverse")
    col3.metric("Incorrect", counts[2])
    col4.metric("Compliance", f"{compliance:.0f}%")

    if counts[1] > 0:
        st.error(f"⚠️  {counts[1]} person(s) detected **WITHOUT** a mask!")
    elif counts[2] > 0:
        st.warning(f"⚠️  {counts[2]} person(s) wearing mask **incorrectly**.")
    elif total > 0:
        st.success("✅  All detected persons are wearing masks correctly.")

    # Compliance trend chart
    if compliance_history is not None and len(compliance_history) > 2:
        df = pd.DataFrame({"Compliance %": list(compliance_history)})
        st.line_chart(df, height=120)

    return compliance


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("😷 Face Mask Detection")
st.caption(
    "Real-time detection using **YOLOv8s** — three classes: "
    "with mask · no mask · incorrect mask"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    weights_input = st.text_input("Model weights", value=DEFAULT_WEIGHTS)
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    imgsz = st.selectbox("Inference size", [320, 480, 640], index=2)

    st.divider()
    st.header("📊 Model Info")
    st.markdown("""
    | Metric | Score |
    |---|---|
    | mAP@0.5 | 0.935 |
    | Recall | 0.892 |
    | Precision | 0.876 |
    | No-mask recall | 0.921 |
    """)

    st.divider()
    st.markdown("**Classes**")
    st.markdown("🟢 With Mask")
    st.markdown("🔴 No Mask ← safety-critical")
    st.markdown("🟠 Incorrect Mask")

    if not Path(weights_input).exists():
        st.error("Weights file not found.")
        st.stop()

    model = load_model(weights_input)
    st.success("Model loaded ✅")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_about = st.tabs(
    ["📷 Image", "🎬 Video", "📹 Webcam (live)", "ℹ️ About"]
)

# ── Image tab ─────────────────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"],
        help="Upload any photo containing faces"
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            results = model(img_bgr, imgsz=imgsz, conf=conf_threshold, verbose=False)
        annotated, counts = draw_boxes(img_bgr, results, conf_threshold)

        col_orig, col_det = st.columns(2)
        col_orig.image(pil_img, caption="Original", use_container_width=True)
        col_det.image(annotated, caption="Detections", use_container_width=True)

        st.divider()
        show_stats(counts)

        # Download annotated image
        annotated_pil = Image.fromarray(annotated)
        buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        annotated_pil.save(buf.name)
        with open(buf.name, "rb") as f:
            st.download_button("⬇️ Download annotated image", f,
                               file_name="detected.jpg", mime="image/jpeg")

# ── Video tab ─────────────────────────────────────────────────────────────────
with tab_video:
    video_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )
    if video_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        stop_btn = st.button("⏹ Stop")

        compliance_history = deque(maxlen=HISTORY_LEN)
        frame_idx = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
            annotated, counts = draw_boxes(frame, results, conf_threshold)

            total = sum(counts.values())
            compliance = (counts[0] / total * 100) if total > 0 else 0.0
            compliance_history.append(compliance)

            frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
            with stats_placeholder.container():
                show_stats(counts, compliance_history)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
            time.sleep(1 / fps_video)

        cap.release()
        progress_bar.progress(1.0)
        st.success("Video processing complete.")

# ── Start background inference thread (module level, runs once) ───────────────
def _start_inference_thread():
    global _wc_thread_started
    if _wc_thread_started:
        return
    _wc_model = load_webcam_model()

    def _inference_worker():
        while True:
            with _wc_lock:
                img = _wc_latest_frame["img"]
            if img is None:
                time.sleep(0.01)
                continue
            small = cv2.resize(img, (640, 640))
            results = _wc_model(small, imgsz=640, conf=0.35, verbose=False)
            scale_x = img.shape[1] / 640
            scale_y = img.shape[0] / 640
            boxes_out = []
            counts = {0: 0, 1: 0, 2: 0}
            for box in results[0].boxes:
                c = float(box.conf)
                if c < 0.35:
                    continue
                cls_id = int(box.cls)
                counts[cls_id] += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_out.append((
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                    cls_id, c
                ))
            with _wc_lock:
                _wc_latest_result["boxes"] = boxes_out
                _wc_latest_result["counts"] = counts

    threading.Thread(target=_inference_worker, daemon=True).start()
    _wc_thread_started = True

_start_inference_thread()

# ── Webcam tab ────────────────────────────────────────────────────────────────
with tab_webcam:
    st.markdown("### Live Webcam Detection")
    st.caption("Allow camera access when prompted by your browser.")

    if "wc_counts" not in st.session_state:
        st.session_state.wc_counts = {0: 0, 1: 0, 2: 0}
    if "wc_history" not in st.session_state:
        st.session_state.wc_history = deque(maxlen=HISTORY_LEN)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Non-blocking: drop latest frame for inference, read last result
        with _wc_lock:
            _wc_latest_frame["img"] = img.copy()
            boxes = list(_wc_latest_result["boxes"])
            counts = dict(_wc_latest_result["counts"])
        # Draw last known boxes immediately — never waits for inference
        annotated = img.copy()
        for (x1, y1, x2, y2, cls_id, conf) in boxes:
            color_bgr = CLASS_COLORS_RGB.get(cls_id, (180, 180, 180))[::-1]
            label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return av.VideoFrame.from_ndarray(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format="rgb24"
        )

    RTC_CONFIGURATION = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }

    webrtc_ctx = webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,
    )

    if webrtc_ctx.state.playing:
        with _wc_lock:
            counts = dict(_wc_latest_result["counts"])
        st.session_state.wc_counts = counts
        total = sum(counts.values())
        compliance = (counts[0] / total * 100) if total > 0 else 0.0
        st.session_state.wc_history.append(compliance)
        show_stats(st.session_state.wc_counts, st.session_state.wc_history)

# ── About tab ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ## About this project

    Real-time face mask detection system built with **YOLOv8s** and deployed with **Streamlit**.

    ### Model
    - Architecture: YOLOv8s (small) — 11M parameters, 28.4 GFLOPs
    - Dataset: Kaggle Andrewmvd face mask dataset (853 images, augmented to ~2,400)
    - Training: 55 epochs on Google Colab T4 GPU (~20 min)

    ### Performance
    | Metric | Score | Target |
    |---|---|---|
    | mAP@0.5 | 0.935 | > 0.90 ✅ |
    | Recall | 0.892 | > 0.85 ✅ |
    | Precision | 0.876 | > 0.88 ≈ |
    | No-mask recall | 0.921 | > 0.92 ≈ |

    ### Classes
    | Class | Color | Description |
    |---|---|---|
    | with_mask | 🟢 Green | Mask worn correctly |
    | without_mask | 🔴 Red | No mask — **safety-critical** |
    | mask_weared_incorrect | 🟠 Orange | Mask worn incorrectly |

    ### Safety design
    The model is tuned to minimise **false negatives on `without_mask`** — missing
    someone without a mask is more costly than a false alarm.

    ### Stack
    `YOLOv8` · `PyTorch` · `OpenCV` · `Albumentations` · `Streamlit`

    ---
    Built as a portfolio project demonstrating end-to-end ML deployment.
    """)
