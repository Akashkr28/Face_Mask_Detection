"""
Sidebar rendering.
render_sidebar() draws the sidebar and returns the user-selected settings
so app.py can pass them into each tab.
"""

from pathlib import Path
from dataclasses import dataclass

import streamlit as st
from ultralytics import YOLO

from core.config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_WEIGHTS,
    MODEL_METRICS,
)
from core.model import load_model


@dataclass
class SidebarConfig:
    """Settings chosen by the user in the sidebar."""
    weights_path: str
    conf: float
    imgsz: int
    model: YOLO


def render_sidebar() -> SidebarConfig | None:
    """
    Render the sidebar UI.

    Returns a SidebarConfig on success, or None (after calling st.stop())
    if the weights file is missing.
    """
    with st.sidebar:
        # ── Logo / title ───────────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <div style='font-size:2.5rem;'>😷</div>
            <div style='font-weight:700; font-size:1.1rem; color:white;'>Mask Detector</div>
            <div style='font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:2px;'>
                YOLOv8s · Real-time
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Model settings ─────────────────────────────────────────────────────
        st.markdown("**⚙️ Model Settings**")
        weights_input = st.text_input(
            "Weights path", value=DEFAULT_WEIGHTS, label_visibility="collapsed"
        )
        conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
        imgsz = st.selectbox("Inference size", [320, 480, 640], index=2)

        st.divider()

        # ── Performance metrics ────────────────────────────────────────────────
        st.markdown("**📊 Model Performance**")
        for label, value, color in MODEL_METRICS:
            st.markdown(f"""
            <div class="sidebar-metric">
                <span class="sidebar-metric-label">{label}</span>
                <span class="sidebar-metric-value" style="color:{color}">{value}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Class legend ───────────────────────────────────────────────────────
        st.markdown("**🎯 Classes**")
        st.markdown("""
        <div style='display:flex; flex-direction:column; gap:6px; margin-top:6px;'>
            <div style='display:flex; align-items:center; gap:8px; font-size:0.85rem;'>
                <div style='width:10px;height:10px;border-radius:50%;background:#00C864;'></div>
                <span style='color:rgba(255,255,255,0.7);'>With Mask</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px; font-size:0.85rem;'>
                <div style='width:10px;height:10px;border-radius:50%;background:#DC3232;'></div>
                <span style='color:rgba(255,255,255,0.7);'>
                    No Mask <span style="color:#DC3232;font-size:0.7rem;">● safety-critical</span>
                </span>
            </div>
            <div style='display:flex; align-items:center; gap:8px; font-size:0.85rem;'>
                <div style='width:10px;height:10px;border-radius:50%;background:#FFA000;'></div>
                <span style='color:rgba(255,255,255,0.7);'>Incorrect Mask</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Validation ─────────────────────────────────────────────────────────
        if not Path(weights_input).exists():
            st.error("⚠️ Weights file not found at the specified path.")
            st.stop()
            return None  # unreachable but satisfies type checker

        model = load_model(weights_input)
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.success("✅ Model ready")

    return SidebarConfig(
        weights_path=weights_input,
        conf=float(conf),
        imgsz=int(imgsz),
        model=model,
    )
