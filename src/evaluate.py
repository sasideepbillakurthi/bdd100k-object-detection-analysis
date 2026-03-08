"""
Evaluation script for object detection on BDD100K.

Supports evaluation for:
1. Faster R-CNN (ResNet50 + FPN)
2. Swin Transformer + Faster R-CNN

Produces:
- quantitative metrics
- per-class precision/recall
- visualization plots
- qualitative prediction samples
"""

import argparse
from pathlib import Path
from typing import Dict

import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from src.config import DETECTION_CLASSES, SAMPLES_DIR, TABLES_DIR, IMAGE_DIR_VAL, LABEL_FILE_VAL
from src.dataset import BDDDetectionDataset
from src.parser import load_annotations
from src.train import BDDTorchDataset, collate_fn, IDX_TO_CLASS

from src.models.swin_faster_rcnn import build_model as build_swin_fasterrcnn
from src.models.faster_rcnn import build_model as build_resnet_fasterrcnn


# ---------------------------------------------------------
# IoU Utility
# ---------------------------------------------------------

def compute_iou_matrix(preds: torch.Tensor, targets: torch.Tensor):

    if preds.numel() == 0 or targets.numel() == 0:
        return torch.zeros((preds.shape[0], targets.shape[0]))

    return box_iou(preds, targets)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_detector(model, dataloader, device, iou_threshold=0.5):

    model.eval()

    stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in DETECTION_CLASSES}

    with torch.no_grad():

        for images, targets in dataloader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):

                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes = output["boxes"]
                pred_labels = output["labels"]
                pred_scores = output["scores"]

                keep = pred_scores >= 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                ious = compute_iou_matrix(pred_boxes, gt_boxes)

                matched_gt = set()

                for i, pred_label in enumerate(pred_labels):

                    cls = IDX_TO_CLASS[pred_label.item()]

                    if ious.shape[1] == 0:
                        stats[cls]["fp"] += 1
                        continue

                    max_iou, gt_idx = ious[i].max(0)

                    if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                        stats[cls]["tp"] += 1
                        matched_gt.add(gt_idx.item())
                    else:
                        stats[cls]["fp"] += 1

                for j, gt_label in enumerate(gt_labels):

                    if j not in matched_gt:

                        cls = IDX_TO_CLASS[gt_label.item()]
                        stats[cls]["fn"] += 1

    return stats


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_precision_recall(stats: Dict):

    rows = []

    for cls, values in stats.items():

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        rows.append(
            {
                "class": cls,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }
        )

    return rows


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def plot_metrics(rows):

    classes = [r["class"] for r in rows]
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]

    plt.figure(figsize=(10, 5))

    x = range(len(classes))

    plt.bar(x, precision, width=0.4, label="Precision")
    plt.bar(x, recall, width=0.4, label="Recall", alpha=0.7)

    plt.xticks(x, classes, rotation=45)

    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall")

    plt.legend()
    plt.tight_layout()

    fig_path = TABLES_DIR / "precision_recall.png"
    plt.savefig(fig_path)

    print(f"[INFO] Saved visualization: {fig_path}")


# ---------------------------------------------------------
# Failure Case Visualization
# ---------------------------------------------------------

def save_failure_cases(dataset, model, device, max_samples=10):

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    with torch.no_grad():

        for image_id, anns in dataset:

            image = dataset.load_image(image_id)

            img_tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.to(device)

            output = model([img_tensor])[0]

            if len(output["boxes"]) == 0 and len(anns) > 0:

                out_path = SAMPLES_DIR / f"missed_{image_id}.jpg"

                cv2.imwrite(str(out_path), image)

                saved += 1

            if saved >= max_samples:
                break


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["fasterrcnn", "swin"],
        required=True
    )

    parser.add_argument("--weights", type=Path, required=True)

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations = load_annotations(LABEL_FILE_VAL)

    dataset = BDDDetectionDataset(IMAGE_DIR_VAL, annotations)

    torch_dataset = BDDTorchDataset(dataset)

    dataloader = DataLoader(
        torch_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if args.model == "fasterrcnn":
        model = build_resnet_fasterrcnn(pretrained=False)

    elif args.model == "swin":
        model = build_swin_fasterrcnn()

    model.load_state_dict(
        torch.load(args.weights, map_location=device)
    )

    model.to(device)

    stats = evaluate_detector(model, dataloader, device)

    rows = compute_precision_recall(stats)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = TABLES_DIR / "evaluation_metrics.csv"

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    plot_metrics(rows)

    save_failure_cases(dataset, model, device)

    print(f"[INFO] Metrics saved to {output_csv}")


if __name__ == "__main__":
    main()