"""
Export a trained YOLOv8 model to ONNX or TFLite for edge deployment.

Usage:
    python scripts/export_model.py --weights runs/detect/mask_detector/weights/best.pt
    python scripts/export_model.py --weights best.pt --format tflite --imgsz 320
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/mask_detector/weights/best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "tflite", "coreml", "openvino"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 (ONNX/CoreML only)")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify,
        dynamic=False,
    )
    print(f"\nExported to: {export_path}")

    if args.format == "onnx":
        import onnx
        m = onnx.load(str(export_path))
        onnx.checker.check_model(m)
        print("ONNX model check passed.")

    # Benchmark: compare PT vs ONNX inference speed
    print("\nBenchmarking inference speed...")
    import time, cv2, numpy as np

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    n_runs = 20

    # PyTorch speed
    model_pt = YOLO(str(weights))
    start = time.time()
    for _ in range(n_runs):
        model_pt(dummy, imgsz=args.imgsz, verbose=False)
    pt_ms = (time.time() - start) / n_runs * 1000

    # ONNX speed
    model_onnx = YOLO(str(export_path))
    start = time.time()
    for _ in range(n_runs):
        model_onnx(dummy, imgsz=args.imgsz, verbose=False)
    onnx_ms = (time.time() - start) / n_runs * 1000

    print(f"PyTorch inference: {pt_ms:.1f} ms/frame  ({1000/pt_ms:.1f} FPS)")
    print(f"ONNX inference:    {onnx_ms:.1f} ms/frame  ({1000/onnx_ms:.1f} FPS)")
    speedup = pt_ms / onnx_ms
    print(f"Speedup: {speedup:.2f}x faster with ONNX")


if __name__ == "__main__":
    main()
