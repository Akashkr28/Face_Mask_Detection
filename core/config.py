"""
Central configuration — all constants live here.
Import from this module everywhere else; never hard-code values in UI files.
"""

# ── Class definitions ──────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = ["With Mask", "No Mask", "Incorrect Mask"]

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 100),    # green
    1: (220, 50, 50),    # red
    2: (255, 160, 0),    # orange
}

CLASS_COLORS_HEX: dict[int, str] = {
    0: "#00C864",
    1: "#DC3232",
    2: "#FFA000",
}

# ── Model paths ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS   = "runs/detect/mask_detector/weights/best.pt"
ONNX_WEIGHTS      = "runs/detect/mask_detector/weights/best.onnx"

# ── Inference defaults ─────────────────────────────────────────────────────────
DEFAULT_CONF      = 0.55
DEFAULT_IMGSZ     = 640
WEBCAM_CONF       = 0.55
WEBCAM_IMGSZ      = 640

# ── UI ─────────────────────────────────────────────────────────────────────────
HISTORY_LEN       = 100          # rolling compliance chart window
COMPLIANCE_HIGH   = 80           # % threshold → green gauge
COMPLIANCE_MID    = 50           # % threshold → orange gauge

# ── Model performance (displayed in sidebar) ───────────────────────────────────
MODEL_METRICS: list[tuple[str, str, str]] = [
    ("mAP@0.5",       "0.935", "#00C864"),
    ("Recall",        "0.892", "#00C864"),
    ("Precision",     "0.876", "#3296FF"),
    ("No-mask Recall","0.921", "#00C864"),
]
