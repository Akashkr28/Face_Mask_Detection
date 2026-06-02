"""
Live webcam tab.
Uses streamlit-webrtc for browser WebRTC streaming.
Inference runs in a background thread (core.inference) so the video
stream is never blocked by model latency.
"""

from collections import deque

import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from core.config import HISTORY_LEN
from core.inference import get_latest_counts, start_inference_thread, video_frame_callback
from ui.components import (
    render_compliance_chart,
    render_count_rows,
    render_gauge,
    section_header,
)

# WebRTC ICE / TURN configuration
_RTC_CONFIG = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
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


def _init_session_state() -> None:
    if "wc_counts"  not in st.session_state:
        st.session_state.wc_counts  = {0: 0, 1: 0, 2: 0}
    if "wc_history" not in st.session_state:
        st.session_state.wc_history = deque(maxlen=HISTORY_LEN)


def render() -> None:
    _init_session_state()

    # Start background thread once per process (idempotent)
    start_inference_thread()

    section_header("Live Webcam Detection")
    st.markdown(
        "<div style='color:rgba(255,255,255,0.4); font-size:0.85rem; margin-bottom:16px;'>"
        "Allow camera access when prompted. Detection runs in a background thread "
        "for smooth, freeze-free video.</div>",
        unsafe_allow_html=True,
    )

    col_wc, col_stats = st.columns([3, 1], gap="medium")

    # ── WebRTC stream ──────────────────────────────────────────────────────────
    with col_wc:
        webrtc_ctx = webrtc_streamer(
            key="mask-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 1920, "min": 1280},
                    "height":    {"ideal": 1080, "min": 720},
                    "frameRate": {"ideal": 30,   "min": 15},
                },
                "audio": False,
            },
            async_processing=True,
            rtc_configuration=_RTC_CONFIG,
        )

    # ── Live stats panel ───────────────────────────────────────────────────────
    with col_stats:
        section_header("Live Stats")

        if webrtc_ctx.state.playing:
            counts = get_latest_counts()
            st.session_state.wc_counts = counts

            total      = sum(counts.values())
            compliance = (counts[0] / total * 100) if total > 0 else 0.0
            st.session_state.wc_history.append(compliance)

            render_gauge(compliance)
            render_count_rows(counts)
            render_compliance_chart(st.session_state.wc_history, height=120)
        else:
            st.markdown("""
            <div style='color:rgba(255,255,255,0.3); font-size:0.85rem;
                        text-align:center; padding:40px 0;'>
                Start the webcam to see live stats
            </div>""", unsafe_allow_html=True)
