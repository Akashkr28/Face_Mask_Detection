"""
Reusable UI components shared across multiple tabs.
All functions emit HTML/Streamlit widgets — no business logic here.
"""

from collections import deque

import pandas as pd
import streamlit as st

from core.config import COMPLIANCE_HIGH, COMPLIANCE_MID


# ── Hero banner ────────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">😷 Face Mask Detection</h1>
        <p class="hero-subtitle">
            Real-time detection using YOLOv8s — three classes: with mask · no mask · incorrect mask
        </p>
        <div class="hero-badges">
            <span class="badge badge-green">mAP@0.5: 93.5%</span>
            <span class="badge badge-blue">YOLOv8s · 11M params</span>
            <span class="badge badge-purple">No-mask Recall: 92.1%</span>
            <span class="badge">Trained on T4 GPU</span>
            <span class="badge">3 Classes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Section header ─────────────────────────────────────────────────────────────
def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Upload zone placeholder ────────────────────────────────────────────────────
def upload_placeholder(icon: str, hint: str) -> None:
    st.markdown(f"""
    <div class="upload-zone">
        <div style='font-size:2.5rem; margin-bottom:10px;'>{icon}</div>
        <div style='color:rgba(255,255,255,0.5); font-size:0.9rem;'>{hint}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Compliance gauge ───────────────────────────────────────────────────────────
def compliance_color(pct: float) -> str:
    if pct >= COMPLIANCE_HIGH:
        return "#00C864"
    if pct >= COMPLIANCE_MID:
        return "#FFA000"
    return "#DC3232"


def render_gauge(pct: float) -> None:
    color = compliance_color(pct)
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-value" style="color:{color}">{pct:.0f}%</div>
        <div class="gauge-label">Compliance</div>
    </div>
    """, unsafe_allow_html=True)


# ── Mini sidebar-style metric rows ────────────────────────────────────────────
def render_count_rows(counts: dict[int, int]) -> None:
    rows = [
        ("🟢 With Mask", counts.get(0, 0), "stat-green"),
        ("🔴 No Mask",   counts.get(1, 0), "stat-red"),
        ("🟠 Incorrect", counts.get(2, 0), "stat-orange"),
    ]
    html = "<div style='margin-top:8px;'>"
    for label, value, cls in rows:
        html += f"""
        <div class="sidebar-metric">
            <span class="sidebar-metric-label">{label}</span>
            <span class="sidebar-metric-value {cls}">{value}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Stat card grid (4-up) ─────────────────────────────────────────────────────
def render_stat_grid(counts: dict[int, int]) -> None:
    total = sum(counts.values())
    pct = (counts.get(0, 0) / total * 100) if total > 0 else 0.0
    color = compliance_color(pct)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <p class="stat-value stat-green">{counts.get(0, 0)}</p>
            <p class="stat-label">With Mask</p>
        </div>
        <div class="stat-card">
            <p class="stat-value stat-red">{counts.get(1, 0)}</p>
            <p class="stat-label">No Mask</p>
        </div>
        <div class="stat-card">
            <p class="stat-value stat-orange">{counts.get(2, 0)}</p>
            <p class="stat-label">Incorrect</p>
        </div>
        <div class="stat-card">
            <p class="stat-value" style="color:{color}">{pct:.0f}%</p>
            <p class="stat-label">Compliance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Alert banners ──────────────────────────────────────────────────────────────
def render_alert(counts: dict[int, int]) -> None:
    no_mask   = counts.get(1, 0)
    incorrect = counts.get(2, 0)
    total     = sum(counts.values())
    if no_mask > 0:
        st.markdown(f"""
        <div class="alert-danger">
            ⚠️ &nbsp; {no_mask} person(s) detected <strong>WITHOUT</strong> a mask — immediate action required!
        </div>""", unsafe_allow_html=True)
    elif incorrect > 0:
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ &nbsp; {incorrect} person(s) wearing mask <strong>incorrectly</strong>.
        </div>""", unsafe_allow_html=True)
    elif total > 0:
        st.markdown("""
        <div class="alert-success">
            ✅ &nbsp; All detected persons are wearing masks correctly.
        </div>""", unsafe_allow_html=True)


# ── Compliance line chart ──────────────────────────────────────────────────────
def render_compliance_chart(history: deque, height: int = 120) -> None:
    if len(history) > 2:
        df = pd.DataFrame({"Compliance %": list(history)})
        st.line_chart(df, height=height, use_container_width=True)


# ── Combined stats block (grid + alert + optional chart) ──────────────────────
def render_stats_block(
    counts: dict[int, int],
    compliance_history: deque | None = None,
) -> float:
    """Render the full stats section. Returns the computed compliance %."""
    render_stat_grid(counts)
    render_alert(counts)
    if compliance_history is not None:
        render_compliance_chart(compliance_history)
    total = sum(counts.values())
    return (counts.get(0, 0) / total * 100) if total > 0 else 0.0
