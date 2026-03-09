"""
Inference script for BDD100K object detection.

Supports:
1. Faster R-CNN
2. Swin + Faster R-CNN

Outputs:
- predicted bounding boxes
- visualization images
"""

import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

from src.config import DETECTION_CLASSES
from src.models.faster_rcnn import build_model as build_resnet_fasterrcnn
from src.models.swin_faster_rcnn import build_model as build_swin_fasterrcnn
from src.train import IDX_TO_CLASS


# ---------------------------------------------------------
# Draw predictions
# ---------------------------------------------------------

def draw_predictions(image, boxes, labels, scores, threshold=0.5):

    for box, label, score in zip(boxes, labels, scores):

        if score < threshold:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = IDX_TO_CLASS[label.item()]

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

        text = f"{class_name}: {score:.2f}"

        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    return image


# ---------------------------------------------------------
# Run inference
# ---------------------------------------------------------

def run_inference(model, image_path, device):

    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(image_rgb).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu()
    scores = outputs["scores"].cpu().numpy()

    return image, boxes, labels, scores


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["fasterrcnn","swin"], required=True)

    parser.add_argument("--weights", type=Path, required=True)

    parser.add_argument("--image_path", type=Path, required=True)

    parser.add_argument("--output", type=Path, default=Path("predictions"))

    parser.add_argument("--score-threshold", type=float, default=0.5)

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)

    # Build model
    if args.model == "fasterrcnn":
        model = build_resnet_fasterrcnn(pretrained=False)

    else:
        model = build_swin_fasterrcnn()

    # Load weights
    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    model.eval()

    

    image, boxes, labels, scores = run_inference(model, args.image_path, device)

    vis = draw_predictions(
        image,
        boxes,
        labels,
        scores,
        args.score_threshold
    )

    output_path = args.output / args.image_path

    cv2.imwrite(str(output_path), vis)

    print(f"[INFO] Saved {output_path}")


if __name__ == "__main__":
    main()