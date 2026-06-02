"""
Download the Andrewmvd face mask dataset from Kaggle.

Prerequisites:
    1. Install kaggle CLI: pip install kaggle
    2. Place your API token at ~/.kaggle/kaggle.json
       (download from https://www.kaggle.com/settings → API → Create New Token)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output data/raw
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading face-mask-detection dataset from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "andrewmvd/face-mask-detection",
         "--path", str(out), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[ERROR] kaggle download failed.")
        print("Make sure ~/.kaggle/kaggle.json exists and kaggle is installed.")
        sys.exit(1)

    # Kaggle dataset puts images and annotations in subdirs — normalize layout
    # Expected after unzip: data/raw/images/ and data/raw/annotations/
    images_dir = out / "images"
    annotations_dir = out / "annotations"
    if not images_dir.exists() or not annotations_dir.exists():
        print("\n[WARN] Expected images/ and annotations/ not found.")
        print(f"Contents of {out}: {list(out.iterdir())}")
        print("Adjust --images_dir and --annotations_dir in convert_voc_to_yolo.py if needed.")
    else:
        n_img = len(list(images_dir.glob("*")))
        n_ann = len(list(annotations_dir.glob("*.xml")))
        print(f"\nDownload complete.")
        print(f"  Images:      {n_img}  → {images_dir}")
        print(f"  Annotations: {n_ann}  → {annotations_dir}")
        print("\nNext step:")
        print("  python scripts/convert_voc_to_yolo.py")


if __name__ == "__main__":
    main()
