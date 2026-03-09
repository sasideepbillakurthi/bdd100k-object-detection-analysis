"""
Comprehensive evaluation script for object detection on BDD100K.

Metrics computed:
- Precision
- Recall
- F1 Score
- Average Precision (AP)
- mean Average Precision (mAP@0.5)
- mean IoU for true positives
- Confusion Matrix
- Precision–Recall curves

Outputs:
- evaluation_metrics.csv
- precision_recall_curve.png
- confusion_matrix.png
- precision_recall_bar.png
- failure examples
"""

import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from src.config import (
    DETECTION_CLASSES,
    SAMPLES_DIR,
    TABLES_DIR,
    IMAGE_DIR_VAL,
    LABEL_FILE_VAL,
)

from src.dataset import BDDDetectionDataset
from src.parser import load_annotations
from src.train import BDDTorchDataset, collate_fn, IDX_TO_CLASS

from src.models.swin_faster_rcnn import build_model as build_swin_fasterrcnn
from src.models.faster_rcnn import build_model as build_resnet_fasterrcnn


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def compute_iou_matrix(pred_boxes, gt_boxes):
    """Compute IoU matrix between predicted and ground-truth boxes."""
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))
    return box_iou(pred_boxes, gt_boxes)


def calculate_ap(recalls, precisions):
    """
    Compute Average Precision using the area under the PR curve.
    Pascal VOC style interpolation.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap


# ---------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------

def evaluate_detector(model, dataloader, device, iou_threshold=0.5):
    """
    Runs inference on validation dataset and collects statistics
    for computing evaluation metrics.
    """

    model.eval()

    stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in DETECTION_CLASSES}

    class_scores = {cls: [] for cls in DETECTION_CLASSES}
    class_matches = {cls: [] for cls in DETECTION_CLASSES}

    tp_ious = []

    num_classes = len(DETECTION_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

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

                keep = pred_scores >= 0.05
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                ious = compute_iou_matrix(pred_boxes, gt_boxes)
                matched_gt = set()

                for i, pred_label in enumerate(pred_labels):

                    pred_class = IDX_TO_CLASS[pred_label.item()]
                    score = pred_scores[i].item()

                    if ious.shape[1] == 0:
                        stats[pred_class]["fp"] += 1
                        class_scores[pred_class].append(score)
                        class_matches[pred_class].append(0)
                        continue

                    max_iou, gt_idx = ious[i].max(0)

                    if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:

                        stats[pred_class]["tp"] += 1
                        tp_ious.append(max_iou.item())

                        matched_gt.add(gt_idx.item())

                        gt_class = IDX_TO_CLASS[gt_labels[gt_idx].item()]

                        gt_idx_cm = DETECTION_CLASSES.index(gt_class)
                        pred_idx_cm = DETECTION_CLASSES.index(pred_class)

                        confusion_matrix[gt_idx_cm, pred_idx_cm] += 1

                        class_scores[pred_class].append(score)
                        class_matches[pred_class].append(1)

                    else:

                        stats[pred_class]["fp"] += 1
                        class_scores[pred_class].append(score)
                        class_matches[pred_class].append(0)

                for j, gt_label in enumerate(gt_labels):
                    if j not in matched_gt:
                        gt_class = IDX_TO_CLASS[gt_label.item()]
                        stats[gt_class]["fn"] += 1

    return stats, confusion_matrix, class_scores, class_matches, tp_ious


# ---------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------

def compute_all_metrics(stats, class_scores, class_matches, tp_ious):

    rows = []

    for cls in DETECTION_CLASSES:

        tp = stats[cls]["tp"]
        fp = stats[cls]["fp"]
        fn = stats[cls]["fn"]

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        scores = np.array(class_scores[cls])
        matches = np.array(class_matches[cls])

        if len(matches) > 0 and np.any(matches == 1):

            order = np.argsort(-scores)
            matches = matches[order]

            tp_cum = np.cumsum(matches)
            fp_cum = np.cumsum(1 - matches)

            total_gt = tp + fn

            precisions = tp_cum / (tp_cum + fp_cum)
            recalls = tp_cum / total_gt

            ap = calculate_ap(recalls, precisions)

        else:
            ap = 0

        rows.append(
            {
                "class": cls,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "ap": ap,
            }
        )

    mAP = np.mean([r["ap"] for r in rows])
    mIoU = np.mean(tp_ious) if tp_ious else 0

    return rows, mAP, mIoU


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def plot_precision_recall_curve(class_scores, class_matches, stats):

    plt.figure(figsize=(8, 8))

    for cls in DETECTION_CLASSES:

        scores = np.array(class_scores[cls])
        matches = np.array(class_matches[cls])

        if len(matches) == 0 or not np.any(matches == 1):
            continue

        order = np.argsort(-scores)
        matches = matches[order]

        tp_cum = np.cumsum(matches)
        fp_cum = np.cumsum(1 - matches)

        total_gt = stats[cls]["tp"] + stats[cls]["fn"]

        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / total_gt

        plt.plot(recall, precision, label=cls)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves per Class")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(TABLES_DIR / "precision_recall_curve.png")


def plot_confusion_matrix(conf_matrix):

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=DETECTION_CLASSES,
        yticklabels=DETECTION_CLASSES,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Detection Confusion Matrix")

    plt.tight_layout()
    plt.savefig(TABLES_DIR / "confusion_matrix.png")


def plot_metrics_bar(rows):

    df = pd.DataFrame(rows)

    df = df.melt(
        id_vars="class",
        value_vars=["precision", "recall", "f1_score", "ap"],
        var_name="metric",
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(data=df, x="class", y="value", hue="metric")

    plt.xticks(rotation=45)
    plt.title("Per-Class Evaluation Metrics")

    plt.tight_layout()
    plt.savefig(TABLES_DIR / "precision_recall_bar.png")


# ---------------------------------------------------------
# Failure Visualization
# ---------------------------------------------------------

def save_failure_cases(dataset, model, device, max_samples=10):

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    with torch.no_grad():

        for image_id, anns in dataset:

            image = dataset.load_image(image_id)

            tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
            tensor = tensor.to(device)

            output = model([tensor])[0]

            if len(output["boxes"]) == 0 and len(anns) > 0:

                cv2.imwrite(
                    str(SAMPLES_DIR / f"missed_{image_id}.jpg"),
                    image,
                )

                saved += 1

            if saved >= max_samples:
                break


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["fasterrcnn", "swin"], required=True)
    parser.add_argument("--weights", type=Path, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations = load_annotations(LABEL_FILE_VAL)

    dataset = BDDDetectionDataset(IMAGE_DIR_VAL, annotations)

    dataloader = DataLoader(
        BDDTorchDataset(dataset),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if args.model == "fasterrcnn":
        model = build_resnet_fasterrcnn(pretrained=False)
    else:
        model = build_swin_fasterrcnn()

    checkpoint = torch.load(args.weights, map_location=device)

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)

    model.to(device)

    stats, conf_matrix, class_scores, class_matches, tp_ious = evaluate_detector(
        model,
        dataloader,
        device,
    )

    rows, mAP, mIoU = compute_all_metrics(
        stats,
        class_scores,
        class_matches,
        tp_ious,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("DETECTION EVALUATION SUMMARY")
    print("==============================")
    print(f"mAP@0.5 : {mAP:.4f}")
    print(f"mIoU    : {mIoU:.4f}")
    print("==============================\n")

    pd.DataFrame(rows).to_csv(
        TABLES_DIR / "evaluation_metrics.csv",
        index=False,
    )

    plot_metrics_bar(rows)
    plot_precision_recall_curve(class_scores, class_matches, stats)
    plot_confusion_matrix(conf_matrix)

    save_failure_cases(dataset, model, device)

    print(f"[INFO] Results saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()