"""
Real-time face mask detection from webcam using OpenCV.

Usage:
    python detect_webcam.py --weights runs/detect/mask_detector/weights/best.pt
    python detect_webcam.py --weights best.pt --conf 0.4 --source 0
    python detect_webcam.py --weights best.pt --source video.mp4

Controls:
    q   — quit
    s   — save current frame as screenshot
    +/- — adjust confidence threshold
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# BGR colors per class
CLASS_COLORS = {
    0: (0, 200, 0),    # with_mask → green
    1: (0, 0, 220),    # without_mask → red
    2: (0, 165, 255),  # mask_weared_incorrect → orange
}
CLASS_NAMES = {
    0: "With Mask",
    1: "No Mask",
    2: "Incorrect Mask",
}

ALERT_CLASSES = {1}  # trigger alert for without_mask


def try_alert():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        # Generate a simple beep via pygame
        sample_rate = 44100
        duration = 0.2
        freq = 880
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        wave = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
    except Exception:
        pass  # alert is best-effort


def draw_detections(frame, results, conf_threshold, alert_triggered):
    counts = {0: 0, 1: 0, 2: 0}

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        counts[cls_id] += 1

    # HUD: stats overlay
    total = sum(counts.values())
    compliant = counts[0]
    compliance = (compliant / total * 100) if total > 0 else 0.0

    hud_lines = [
        f"With Mask:    {counts[0]}",
        f"No Mask:      {counts[1]}",
        f"Incorrect:    {counts[2]}",
        f"Compliance:   {compliance:.0f}%",
        f"Conf thresh:  {conf_threshold:.2f}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if alert_triggered:
        cv2.putText(frame, "! NO MASK DETECTED !", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/mask_detector/weights/best.pt")
    parser.add_argument("--source", default="0",
                        help="Camera index (0,1,...) or path to video file")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.fuse()  # fuse Conv+BN layers for faster inference

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    conf = args.conf
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    screenshot_n = 0
    last_alert_time = 0
    alert_cooldown = 3.0  # seconds between audio alerts

    print("Running — press 'q' to quit, 's' to screenshot, '+'/'-' to adjust confidence")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=args.imgsz, conf=conf, verbose=False)

        alert_classes_present = any(
            int(box.cls) in ALERT_CLASSES and float(box.conf) >= conf
            for box in results[0].boxes
        )
        now = time.time()
        if alert_classes_present and (now - last_alert_time) > alert_cooldown:
            try_alert()
            last_alert_time = now

        counts = draw_detections(frame, results, conf, alert_classes_present)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow("Face Mask Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            screenshot_n += 1
        elif key == ord("+") or key == ord("="):
            conf = min(conf + 0.05, 0.95)
            print(f"Confidence → {conf:.2f}")
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)
            print(f"Confidence → {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
