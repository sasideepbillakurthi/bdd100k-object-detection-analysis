"""
Evaluation script for object detection on BDD100K.

Supports evaluation for:
1. Faster R-CNN (ResNet50 + FPN)
2. Swin Transformer + Faster R-CNN

Produces:
- quantitative metrics
- per-class precision/recall
- precision-recall curve
- confusion matrix
- qualitative prediction samples
"""

import argparse
from pathlib import Path
from typing import Dict

import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
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

    num_classes = len(DETECTION_CLASSES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    all_scores = []
    all_matches = []

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
                pred_scores = pred_scores[keep]

                ious = compute_iou_matrix(pred_boxes, gt_boxes)

                matched_gt = set()

                for i, pred_label in enumerate(pred_labels):

                    cls = IDX_TO_CLASS[pred_label.item()]
                    score = pred_scores[i].item()

                    if ious.shape[1] == 0:
                        stats[cls]["fp"] += 1
                        all_scores.append(score)
                        all_matches.append(0)
                        continue

                    max_iou, gt_idx = ious[i].max(0)

                    if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:

                        stats[cls]["tp"] += 1
                        matched_gt.add(gt_idx.item())

                        gt_cls_idx = gt_labels[gt_idx].item() - 1
                        pred_cls_idx = pred_label.item() - 1

                        conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                        all_scores.append(score)
                        all_matches.append(1)

                    else:

                        stats[cls]["fp"] += 1
                        all_scores.append(score)
                        all_matches.append(0)

                for j, gt_label in enumerate(gt_labels):

                    if j not in matched_gt:

                        cls = IDX_TO_CLASS[gt_label.item()]
                        stats[cls]["fn"] += 1

    return stats, conf_matrix, torch.tensor(all_scores), torch.tensor(all_matches)


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
# Precision Recall Curve
# ---------------------------------------------------------

def plot_precision_recall_curve(scores, matches):

    thresholds = torch.linspace(0, 1, 50)

    precisions = []
    recalls = []

    for t in thresholds:

        keep = scores >= t

        tp = (matches[keep] == 1).sum().item()
        fp = (matches[keep] == 0).sum().item()
        fn = (matches == 1).sum().item() - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(6,6))

    plt.plot(recalls, precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.grid(True)

    plt.savefig(TABLES_DIR / "precision_recall_curve.png")


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

def plot_confusion_matrix(conf_matrix):

    plt.figure(figsize=(10,8))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        xticklabels=DETECTION_CLASSES,
        yticklabels=DETECTION_CLASSES,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    plt.title("Detection Confusion Matrix")

    plt.tight_layout()

    plt.savefig(TABLES_DIR / "confusion_matrix.png")


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

    plt.savefig(TABLES_DIR / "precision_recall.png")


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
    else:
        model = build_swin_fasterrcnn()

    model.load_state_dict(
        torch.load(args.weights, map_location=device)
    )

    model.to(device)

    stats, conf_matrix, scores, matches = evaluate_detector(model, dataloader, device)

    rows = compute_precision_recall(stats)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "evaluation_metrics.csv", index=False)

    plot_metrics(rows)

    plot_precision_recall_curve(scores, matches)

    plot_confusion_matrix(conf_matrix)

    save_failure_cases(dataset, model, device)

    print("[INFO] Evaluation complete")


if __name__ == "__main__":
    main()