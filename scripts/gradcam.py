"""
Grad-CAM visualization — shows which image regions drive the model's predictions.
Uses pytorch-grad-cam under the hood.

pip install grad-cam

Usage:
    python scripts/gradcam.py --weights best.pt --image path/to/image.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    raise ImportError("Install grad-cam: pip install grad-cam")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/mask_detector/weights/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--class_id", type=int, default=1,
                        help="Class to visualize (0=with_mask, 1=without_mask, 2=incorrect)")
    parser.add_argument("--output", default="gradcam_output.jpg")
    args = parser.parse_args()

    model = YOLO(args.weights)
    backbone = model.model.model  # access underlying nn.Module

    # Target the last C2f block in the backbone
    target_layers = [backbone[-2]]

    img_path = Path(args.image)
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_float = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    cam = GradCAM(model=backbone, target_layers=target_layers)
    targets = [ClassifierOutputTarget(args.class_id)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    out_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
