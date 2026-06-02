"""
Inference helpers:
  - draw_boxes()           – annotate a BGR frame and return RGB + counts
  - WebcamInferenceThread  – singleton background thread for live webcam
  - video_frame_callback() – streamlit-webrtc frame callback (non-blocking)
"""

import threading
import time

import av
import cv2
import numpy as np

from core.config import CLASS_COLORS_RGB, CLASS_NAMES, WEBCAM_CONF, WEBCAM_IMGSZ
from core.model import load_webcam_model

# ── Type alias ─────────────────────────────────────────────────────────────────
BBox = tuple[int, int, int, int, int, float]  # x1, y1, x2, y2, cls_id, conf


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(
    image_bgr: np.ndarray,
    results,
    conf_threshold: float,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Draw bounding boxes + labels on *image_bgr*.

    Returns:
        annotated_rgb  – RGB numpy array ready for st.image()
        counts         – {cls_id: count} for classes 0-2
    """
    frame = image_bgr.copy()
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls)
        counts[cls_id] = counts.get(cls_id, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color_bgr = CLASS_COLORS_RGB.get(cls_id, (180, 180, 180))[::-1]
        label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(
            frame, label, (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), counts


# ── Webcam background inference thread ────────────────────────────────────────
_lock = threading.Lock()
_latest_frame: dict = {"img": None}
_latest_result: dict = {"boxes": [], "counts": {0: 0, 1: 0, 2: 0}}
_thread_started = False


def _inference_worker() -> None:
    """Continuously reads the latest webcam frame and runs detection."""
    model = load_webcam_model()
    while True:
        with _lock:
            img = _latest_frame["img"]
        if img is None:
            time.sleep(0.01)
            continue

        small = cv2.resize(img, (WEBCAM_IMGSZ, WEBCAM_IMGSZ))
        results = model(small, imgsz=WEBCAM_IMGSZ, conf=WEBCAM_CONF, verbose=False)

        scale_x = img.shape[1] / WEBCAM_IMGSZ
        scale_y = img.shape[0] / WEBCAM_IMGSZ
        boxes_out: list[BBox] = []
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

        for box in results[0].boxes:
            c = float(box.conf)
            if c < WEBCAM_CONF:
                continue
            cls_id = int(box.cls)
            counts[cls_id] = counts.get(cls_id, 0) + 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes_out.append((
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y),
                cls_id, c,
            ))

        with _lock:
            _latest_result["boxes"] = boxes_out
            _latest_result["counts"] = counts


def start_inference_thread() -> None:
    """Start the background inference thread (idempotent — safe to call multiple times)."""
    global _thread_started
    if _thread_started:
        return
    threading.Thread(target=_inference_worker, daemon=True).start()
    _thread_started = True


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    streamlit-webrtc callback.  Never blocks on inference — just:
      1. Deposits the latest frame for the background thread.
      2. Draws the most-recently computed boxes and returns immediately.
    """
    img = frame.to_ndarray(format="bgr24")

    with _lock:
        _latest_frame["img"] = img.copy()
        boxes = list(_latest_result["boxes"])

    annotated = img.copy()
    for (x1, y1, x2, y2, cls_id, conf) in boxes:
        color_bgr = CLASS_COLORS_RGB.get(cls_id, (180, 180, 180))[::-1]
        label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(
            annotated, label, (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return av.VideoFrame.from_ndarray(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format="rgb24"
    )


def get_latest_counts() -> dict[int, int]:
    """Return the most-recently computed detection counts (thread-safe)."""
    with _lock:
        return dict(_latest_result["counts"])
