"""
Training script for object detection on BDD100K using Faster R-CNN.

This script demonstrates how to load the BDD100K dataset, prepare
targets, and train a pretrained detection model for a small subset
and limited epochs.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from src.config import DETECTION_CLASSES
from src.parser import Annotation, load_annotations
from src.dataset import BDDDetectionDataset


CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(DETECTION_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class BDDTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for BDD100K object detection.
    """

    def __init__(
        self,
        dataset: BDDDetectionDataset,
        subset_ratio: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.image_ids = dataset.image_ids

        if subset_ratio < 1.0:
            subset_size = int(len(self.image_ids) * subset_ratio)
            self.image_ids = random.sample(self.image_ids, subset_size)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = self.dataset.load_image(image_id)
        annotations = self.dataset.get_annotations(image_id)

        # Convert image to tensor
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
    """Custom collate function for detection models."""
    return tuple(zip(*batch))


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
) -> None:
    """Train model for one epoch."""
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print("[INFO] One epoch training completed")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on BDD100K"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to BDD100K label JSON",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=0.02,
        help="Fraction of dataset to use (e.g., 0.02 = 2%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model.pth"),
        help="Path to save trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    annotations = load_annotations(args.labels)
    dataset = BDDDetectionDataset(args.images, annotations)

    torch_dataset = BDDTorchDataset(
        dataset=dataset,
        subset_ratio=args.subset,
    )

    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    for epoch in range(args.epochs):
        print(f"[INFO] Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, dataloader, optimizer, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[INFO] Model saved to {args.output}")


if __name__ == "__main__":
    main()