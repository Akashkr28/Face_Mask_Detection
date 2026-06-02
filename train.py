"""
Train YOLOv8 on the face mask dataset.

Usage:
    python train.py
    python train.py --model yolov8m.pt --epochs 100 --imgsz 640 --batch 16
    python train.py --no-wandb
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base model: yolov8n/s/m/l/x.pt")
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="mask_detector")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="face-mask-detection", config=vars(args))
        except ImportError:
            print("[WARN] wandb not installed — training without experiment tracking")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Class weights: upweight without_mask (class 1) to reduce false negatives
        # on the safety-critical class. Adjust based on your class distribution.
        cls=1.5,
        # Augmentation (on top of offline augmentation)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Logging
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    print(f"\nTraining complete.")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
