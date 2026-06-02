import streamlit as st

from ui.styles import inject_styles
from ui.sidebar import render_sidebar
from ui.components import render_hero
from ui.tabs import image_tab, video_tab, webcam_tab, about_tab

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ──────────────────────────────────────────────────────────
inject_styles()

# ── Sidebar: settings + model ──────────────────────────────────────────────────
cfg = render_sidebar()   # returns None + calls st.stop() if weights missing

# ── Hero banner ────────────────────────────────────────────────────────────────
render_hero()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_about = st.tabs([
    "📷  Image",
    "🎬  Video",
    "📹  Webcam (live)",
    "ℹ️  About",
])

with tab_image:
    image_tab.render(model=cfg.model, conf=cfg.conf, imgsz=cfg.imgsz)

with tab_video:
    video_tab.render(model=cfg.model, conf=cfg.conf, imgsz=cfg.imgsz)

with tab_webcam:
    webcam_tab.render()

with tab_about:
    about_tab.render()
