"""
Split a converted dataset into train/val/test.

Usage:
    python scripts/split_dataset.py \
        --source data/converted \
        --output data/yolo \
        --train 0.7 --val 0.2 --test 0.1 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/converted")
    parser.add_argument("--output", default="data/yolo")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Splits must sum to 1.0"

    source = Path(args.source)
    output = Path(args.output)

    images = sorted((source / "images").glob("*.*"))
    random.seed(args.seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, files in splits.items():
        img_out = output / "images" / split
        lbl_out = output / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = source / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)
            else:
                # Create empty label file for images with no annotations
                (lbl_out / (img_path.stem + ".txt")).touch()

        print(f"{split:5s}: {len(files)} images → {img_out}")

    print(f"\nTotal: {n} images split into {output.resolve()}")


if __name__ == "__main__":
    main()
