"""
Global CSS injected once at app startup via inject_styles().
Keep all visual styling here — no inline CSS in tab/component files.
"""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero banner ────────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(100,100,255,0.15), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
}
.badge-green  { background: rgba(0,200,100,0.15);  border-color: rgba(0,200,100,0.3);  color: #00C864; }
.badge-blue   { background: rgba(50,150,255,0.15); border-color: rgba(50,150,255,0.3); color: #3296FF; }
.badge-purple { background: rgba(180,100,255,0.15);border-color: rgba(180,100,255,0.3);color: #B464FF; }

/* ── Stat cards ─────────────────────────────────────────────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0;
}
.stat-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-green  { color: #00C864; }
.stat-red    { color: #DC3232; }
.stat-orange { color: #FFA000; }
.stat-white  { color: #ffffff; }

/* ── Alert banners ──────────────────────────────────────────────────────────── */
.alert-danger {
    background: linear-gradient(90deg, rgba(220,50,50,0.15), rgba(220,50,50,0.05));
    border-left: 4px solid #DC3232;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ff6b6b;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 12px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-warning {
    background: linear-gradient(90deg, rgba(255,160,0,0.15), rgba(255,160,0,0.05));
    border-left: 4px solid #FFA000;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ffb74d;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 12px 0;
}
.alert-success {
    background: linear-gradient(90deg, rgba(0,200,100,0.15), rgba(0,200,100,0.05));
    border-left: 4px solid #00C864;
    border-radius: 8px;
    padding: 14px 18px;
    color: #00C864;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 12px 0;
}

/* ── Compliance gauge ───────────────────────────────────────────────────────── */
.gauge-container {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 14px 0;
}
.gauge-value {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}
.gauge-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0e0e1a;
    padding: 6px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    color: white !important;
}

/* ── Upload zone ────────────────────────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed #2a2a4a;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    background: #0e0e1a;
    transition: border-color 0.2s;
}

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0e0e1a;
    border-right: 1px solid #1a1a2e;
}
.sidebar-metric {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric-label { color: rgba(255,255,255,0.5); font-size: 0.82rem; }
.sidebar-metric-value { font-weight: 700; font-size: 0.95rem; }

/* ── Section headers ────────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2a2a4a, transparent);
    margin-left: 10px;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e0e1a; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }

/* ── Download button ────────────────────────────────────────────────────────── */
.stDownloadButton button {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    border: 1px solid #4a4a8a !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 500 !important;
}
</style>
"""


def inject_styles() -> None:
    """Inject global CSS into the Streamlit page. Call once at startup."""
    st.markdown(_CSS, unsafe_allow_html=True)
