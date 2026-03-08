import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from src.config import (
    DETECTION_CLASSES,
    IMAGE_DIR_TRAIN,
    LABEL_FILE_TRAIN,
)

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

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device):

    model.train()
    total_loss = 0

    for images, targets in dataloader:

        images = [img.to(device) for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)

    print(f"[INFO] Avg Loss: {avg_loss:.4f}")


# ---------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------

def train_detector(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -------------------------------------------------
    # Dataset paths from config
    # -------------------------------------------------

    image_dir = IMAGE_DIR_TRAIN
    label_file = LABEL_FILE_TRAIN

    print(f"[INFO] Images: {image_dir}")
    print(f"[INFO] Labels: {label_file}")

    annotations = load_annotations(label_file)

    dataset = BDDDetectionDataset(image_dir, annotations)

    torch_dataset = BDDTorchDataset(dataset, subset_ratio=args.subset)

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

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):

        print(f"[INFO] Epoch {epoch+1}/{args.epochs}")

        train_one_epoch(
            model,
            dataloader,
            optimizer,
            device
        )

        scheduler.step()

        print(f"[INFO] LR: {scheduler.get_last_lr()[0]:.6f}")

        checkpoint_path = checkpoint_dir / f"{args.model}_epoch_{epoch+1}.pth"

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, checkpoint_path)

        print(f"[INFO] Saved checkpoint: {checkpoint_path}")

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
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--subset",
        type=float,
        default=1,
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