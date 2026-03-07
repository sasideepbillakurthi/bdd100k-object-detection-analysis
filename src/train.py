"""
Training script for BDD100K object detection.

Supports two models:
1. Faster R-CNN (ResNet50 + FPN)
2. Swin Transformer + Faster R-CNN

Example
-------

Train ResNet FasterRCNN
python -m src.train --model fasterrcnn --images data/images --labels labels.json

Train Swin FasterRCNN
python -m src.train --model swin --images data/images --labels labels.json
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

from src.models.faster_rcnn import build_model as build_resnet_fasterrcnn
from src.models.swin_faster_rcnn import build_model as build_swin_fasterrcnn


# ---------------------------------------------------------
# Class mappings
# ---------------------------------------------------------

CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(DETECTION_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ---------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

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
# Training pipeline
# ---------------------------------------------------------

def train_detector(args):

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
        num_workers=4,
        pin_memory=True
    )

    # -------------------------------------------------
    # Model selection
    # -------------------------------------------------

    if args.model == "fasterrcnn":
        model = build_resnet_fasterrcnn(len(DETECTION_CLASSES))

    elif args.model == "swin":
        model = build_swin_fasterrcnn()

    else:
        raise ValueError("Unsupported model")

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

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
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="BDD100K training pipeline"
    )

    parser.add_argument(
        "--model",
        choices=["fasterrcnn", "swin"],
        required=True,
        help="Model type"
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Image directory"
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Annotation JSON"
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
        default="outputs/models/model.pth"
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    print(f"[INFO] Training {args.model}")

    train_detector(args)


if __name__ == "__main__":
    main()