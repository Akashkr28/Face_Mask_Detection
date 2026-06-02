"""
Offline augmentation pass to expand the training set.
Reads from data/yolo/images/train + labels/train,
writes augmented copies back to the same directories.

Usage:
    python scripts/augment.py --multiplier 3 --seed 42
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


AUGMENT_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Rotate(limit=15, p=0.4),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.RandomShadow(p=0.2),
        A.CLAHE(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
)


def load_yolo_labels(label_path: Path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return [], []
    lines = label_path.read_text().strip().splitlines()
    class_ids, bboxes = [], []
    for line in lines:
        parts = line.split()
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes


def save_yolo_labels(label_path: Path, class_ids, bboxes):
    lines = []
    for cls_id, box in zip(class_ids, bboxes):
        lines.append(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    label_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", default="data/yolo/images/train")
    parser.add_argument("--train_labels", default="data/yolo/labels/train")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="How many augmented copies to generate per original image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = Path(args.train_images)
    lbl_dir = Path(args.train_labels)
    images = sorted(img_dir.glob("*.*"))
    original_count = len(images)

    print(f"Augmenting {original_count} training images × {args.multiplier} copies")
    generated = 0

    for img_path in images:
        if "_aug" in img_path.stem:
            continue  # skip already-augmented files on re-run

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_ids, bboxes = load_yolo_labels(lbl_dir / (img_path.stem + ".txt"))

        for i in range(args.multiplier):
            try:
                result = AUGMENT_PIPELINE(
                    image=image,
                    bboxes=bboxes if bboxes else [],
                    class_labels=class_ids if class_ids else [],
                )
            except Exception as e:
                print(f"  [WARN] Augmentation failed for {img_path.name}: {e}")
                continue

            aug_stem = f"{img_path.stem}_aug{i}"
            aug_img_path = img_dir / (aug_stem + img_path.suffix)
            aug_lbl_path = lbl_dir / (aug_stem + ".txt")

            out_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), out_img)
            save_yolo_labels(aug_lbl_path, result["class_labels"], result["bboxes"])
            generated += 1

    print(f"Done. Generated {generated} augmented images. Total training set: {original_count + generated}")


if __name__ == "__main__":
    main()
