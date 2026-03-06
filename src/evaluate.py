"""
Evaluation script for object detection on BDD100K.

Supports evaluation for:
1. Faster R-CNN
2. YOLOv8

Produces:
- quantitative metrics
- per-class precision/recall
- visualization plots
- qualitative prediction samples
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from ultralytics import YOLO

from src.config import DETECTION_CLASSES, SAMPLES_DIR, TABLES_DIR
from src.dataset import BDDDetectionDataset
from src.parser import load_annotations
from src.train import BDDTorchDataset, collate_fn, IDX_TO_CLASS


# ---------------------------------------------------------
# FasterRCNN evaluation
# ---------------------------------------------------------

def compute_iou_matrix(preds: torch.Tensor, targets: torch.Tensor):
    if preds.numel() == 0 or targets.numel() == 0:
        return torch.zeros((preds.shape[0], targets.shape[0]))
    return box_iou(preds, targets)


def evaluate_fasterrcnn(model, dataloader, device, iou_threshold=0.5):

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
# YOLO evaluation
# ---------------------------------------------------------

def evaluate_yolo(model_path, data_yaml):

    model = YOLO(model_path)

    metrics = model.val(data=data_yaml)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    return results


# ---------------------------------------------------------
# Metrics processing
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
# Qualitative samples
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

                out_path = SAMPLES_DIR / f"missed_{image_id}"

                cv2.imwrite(str(out_path), image)

                saved += 1

            if saved >= max_samples:
                break


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["fasterrcnn", "yolov8"], required=True)

    parser.add_argument("--labels", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--weights", type=Path)

    parser.add_argument("--data-yaml", type=str)

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    if args.model == "fasterrcnn":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        annotations = load_annotations(args.labels)

        dataset = BDDDetectionDataset(args.images, annotations)

        torch_dataset = BDDTorchDataset(dataset)

        dataloader = DataLoader(
            torch_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = fasterrcnn_resnet50_fpn(pretrained=False)

        model.load_state_dict(torch.load(args.weights, map_location=device))

        model.to(device)

        stats = evaluate_fasterrcnn(model, dataloader, device)

        rows = compute_precision_recall(stats)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        output_csv = TABLES_DIR / "evaluation_metrics.csv"

        pd.DataFrame(rows).to_csv(output_csv, index=False)

        plot_metrics(rows)

        save_failure_cases(dataset, model, device)

        print(f"[INFO] Metrics saved to {output_csv}")

    elif args.model == "yolov8":

        results = evaluate_yolo(args.weights, args.data_yaml)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        metrics_file = TABLES_DIR / "yolo_metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=4)

        print("[INFO] YOLO evaluation complete")
        print(results)


if __name__ == "__main__":
    main()