"""
Evaluate a trained model on the test set.
Prints per-class metrics and generates a confusion matrix.

Usage:
    python evaluate.py --weights runs/detect/mask_detector/weights/best.pt
    python evaluate.py --weights best.pt --conf 0.3 --iou 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
TARGET_METRICS = {
    "mAP50": 0.90,
    "mAP50-95": 0.65,
    "precision": 0.88,
    "recall": 0.85,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/mask_detector/weights/best.pt")
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


def print_metrics_table(metrics):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "Class": name,
            "Precision": f"{metrics.box.p[i]:.3f}",
            "Recall": f"{metrics.box.r[i]:.3f}",
            "mAP@0.5": f"{metrics.box.ap50[i]:.3f}",
            "mAP@0.5:0.95": f"{metrics.box.ap[i]:.3f}",
        })
    rows.append({
        "Class": "ALL (mean)",
        "Precision": f"{metrics.box.mp:.3f}",
        "Recall": f"{metrics.box.mr:.3f}",
        "mAP@0.5": f"{metrics.box.map50:.3f}",
        "mAP@0.5:0.95": f"{metrics.box.map:.3f}",
    })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "-" * 60)
    print("TARGET vs ACHIEVED")
    print("-" * 60)
    achieved = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }
    for metric, target in TARGET_METRICS.items():
        value = achieved[metric]
        status = "PASS" if value >= target else "FAIL"
        print(f"  {metric:15s}: {value:.3f}  (target {target:.2f})  [{status}]")

    # Safety-critical metric: without_mask recall (class index 1)
    wm_recall = metrics.box.r[1]
    wm_target = 0.92
    status = "PASS" if wm_recall >= wm_target else "FAIL"
    print(f"  {'without_mask_R':15s}: {wm_recall:.3f}  (target {wm_target:.2f})  [{status}]  ← safety-critical")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    print(f"Evaluating {weights_path} on {args.split} split...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        plots=True,
        save_json=True,
    )

    print_metrics_table(metrics)

    # Bar chart: per-class recall
    recalls = [metrics.box.r[i] for i in range(len(CLASS_NAMES))]
    colors = ["green" if r >= 0.85 else "red" for r in recalls]
    colors[1] = "green" if recalls[1] >= 0.92 else "red"  # stricter target for without_mask

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, recalls, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0.85, color="orange", linestyle="--", label="Recall target (0.85)")
    ax.axhline(0.92, color="red", linestyle=":", label="without_mask target (0.92)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class Recall")
    ax.legend()
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=10)
    out_path = Path("runs/eval_recall.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Recall chart saved to {out_path}")


if __name__ == "__main__":
    main()
