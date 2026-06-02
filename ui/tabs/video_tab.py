"""
Video detection tab.
Processes an uploaded video frame-by-frame, showing a live annotated
stream alongside a real-time compliance gauge and trend chart.
"""

import tempfile
import time
from collections import deque

import cv2
import streamlit as st
from ultralytics import YOLO

from core.config import HISTORY_LEN
from core.inference import draw_boxes
from ui.components import (
    compliance_color,
    render_compliance_chart,
    render_count_rows,
    render_gauge,
    section_header,
    upload_placeholder,
)


def render(model: YOLO, conf: float, imgsz: int) -> None:
    section_header("Upload a Video")

    video_file = st.file_uploader(
        "Drop a video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if not video_file:
        upload_placeholder(
            icon="🎬",
            hint="Drag & drop a video, or click <b>Browse files</b><br>"
                 "Supports MP4, AVI, MOV, MKV",
        )
        return

    # ── Save to temp file so OpenCV can open it ────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25

    # ── Layout: stream | live stats ───────────────────────────────────────────
    col_vid, col_stats = st.columns([3, 1], gap="medium")

    with col_vid:
        frame_ph = st.empty()

    with col_stats:
        section_header("Live Stats")
        gauge_ph  = st.empty()
        counts_ph = st.empty()
        chart_ph  = st.empty()

    progress_bar        = st.progress(0)
    col_stop, col_info  = st.columns([1, 3])

    with col_stop:
        stop_btn = st.button("⏹  Stop", use_container_width=True)
    with col_info:
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.4); font-size:0.8rem; padding-top:8px;'>"
            f"Total frames: {total_frames} · FPS: {fps_video:.0f}</div>",
            unsafe_allow_html=True,
        )

    # ── Frame loop ─────────────────────────────────────────────────────────────
    compliance_history: deque = deque(maxlen=HISTORY_LEN)
    frame_idx = 0

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        results              = model(frame, imgsz=imgsz, conf=conf, verbose=False)
        annotated_rgb, counts = draw_boxes(frame, results, conf)

        total      = sum(counts.values())
        compliance = (counts[0] / total * 100) if total > 0 else 0.0
        compliance_history.append(compliance)

        # update stream
        frame_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

        # update stats panel
        with gauge_ph.container():
            render_gauge(compliance)
        with counts_ph.container():
            render_count_rows(counts)
        with chart_ph.container():
            render_compliance_chart(compliance_history, height=120)

        frame_idx += 1
        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
        time.sleep(1 / fps_video)

    cap.release()
    progress_bar.progress(1.0)
    st.success("✅ Video processing complete.")
