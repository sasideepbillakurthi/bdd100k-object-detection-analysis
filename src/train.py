"""
Unified training script for BDD100K object detection.

Supports two models:
1. Faster R-CNN (ResNet50 + FPN)
2. YOLOv8 (Ultralytics)

Example
-------

Train FasterRCNN
python src/train.py --model fasterrcnn --images data/images --labels labels.json

Train YOLOv8
python src/train.py --model yolov8 --data-yaml data/bdd.yaml
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from src.config import DETECTION_CLASSES
from src.parser import load_annotations
from src.dataset import BDDDetectionDataset
from src.models.faster_rcnn import build_model
from src.models.yolov8 import train_model as train_yolo


# ---------------------------------------------------------
# FasterRCNN utilities
# ---------------------------------------------------------

CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(DETECTION_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class BDDTorchDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset wrapper converting BDD dataset
    into tensors for FasterRCNN.
    """

    def __init__(self, dataset, subset_ratio=1.0):

        self.dataset = dataset
        self.image_ids = dataset.image_ids

        if subset_ratio < 1.0:
            subset_size = int(len(self.image_ids) * subset_ratio)
            self.image_ids = random.sample(self.image_ids, subset_size)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image = self.dataset.load_image(image_id)
        annotations = self.dataset.get_annotations(image_id)

        image = F.to_tensor(image)

        boxes = []
        labels = []

        for ann in annotations:

            bbox = ann.bbox

            boxes.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            labels.append(CLASS_TO_IDX[ann.category])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    for images, targets in dataloader:

        images = [img.to(device) for img in images]

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print("[INFO] Epoch completed")


# ---------------------------------------------------------
# FasterRCNN training pipeline
# ---------------------------------------------------------

def train_fasterrcnn(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")

    annotations = load_annotations(args.labels)

    dataset = BDDDetectionDataset(args.images, annotations)

    torch_dataset = BDDTorchDataset(
        dataset,
        subset_ratio=args.subset
    )

    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = build_model(len(DETECTION_CLASSES))

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    for epoch in range(args.epochs):

        print(f"[INFO] Epoch {epoch+1}/{args.epochs}")

        train_one_epoch(
            model,
            dataloader,
            optimizer,
            device
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), args.output)

    print(f"[INFO] Model saved to {args.output}")


# ---------------------------------------------------------
# YOLOv8 pipeline
# ---------------------------------------------------------

def train_yolov8(args):

    train_yolo(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="BDD100K training pipeline"
    )

    parser.add_argument(
        "--model",
        choices=["fasterrcnn", "yolov8"],
        required=True,
        help="Model type"
    )

    parser.add_argument(
        "--images",
        type=Path,
        help="Image directory"
    )

    parser.add_argument(
        "--labels",
        type=Path,
        help="Annotation JSON"
    )

    parser.add_argument(
        "--data-yaml",
        type=str,
        default="data/bdd.yaml",
        help="YOLO dataset config"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--subset",
        type=float,
        default=0.02,
        help="Dataset fraction"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/fasterrcnn.pth"
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    if args.model == "fasterrcnn":

        print("[INFO] Training FasterRCNN")

        train_fasterrcnn(args)

    elif args.model == "yolov8":

        print("[INFO] Training YOLOv8")

        train_yolov8(args)


if __name__ == "__main__":
    main()