"""
Convert Pascal VOC XML annotations to YOLO format.

Usage:
    python scripts/convert_voc_to_yolo.py \
        --images_dir data/raw/images \
        --annotations_dir data/raw/annotations \
        --output_dir data/converted
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2,
}


def convert_box(size, box):
    """Convert VOC (xmin,ymin,xmax,ymax) to YOLO (cx,cy,w,h) normalized."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(xml_path: Path, output_label_path: Path) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            print(f"  [WARN] Unknown class '{cls_name}' in {xml_path.name} — skipping object")
            continue
        cls_id = CLASS_MAP[cls_name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        cx, cy, w, h = convert_box((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not lines:
        return False

    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text("\n".join(lines))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/raw/images")
    parser.add_argument("--annotations_dir", default="data/raw/annotations")
    parser.add_argument("--output_dir", default="data/converted")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    xml_files = list(annotations_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")

    converted, skipped = 0, 0
    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  [SKIP] No image found for {xml_path.name}")
            skipped += 1
            continue

        label_path = out_labels / (stem + ".txt")
        ok = convert_annotation(xml_path, label_path)
        if ok:
            shutil.copy2(img_path, out_images / img_path.name)
            converted += 1
        else:
            skipped += 1

    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
