"""
Convert BDD100K annotations to YOLO format.

This script converts BDD JSON annotations into YOLO label files.
Each image gets a corresponding `.txt` file with normalized
bounding box coordinates.

Usage:
python src/scripts/convert_to_yolo.py \
    --images data/images/train \
    --annotations data/labels/train.json \
    --output data/yolo_labels/train
"""

import argparse
import json
from pathlib import Path
from PIL import Image

from src.config import DETECTION_CLASSES


CLASS_TO_ID = {c: i for i, c in enumerate(DETECTION_CLASSES)}


def convert_dataset(images_dir, annotation_file, output_dir):

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file) as f:
        annotations = json.load(f)

    for item in annotations:

        image_name = item["name"]
        image_path = images_dir / image_name

        if not image_path.exists():
            continue

        width, height = Image.open(image_path).size

        label_file = output_dir / image_name.replace(".jpg", ".txt")

        lines = []

        if "labels" not in item:
            label_file.write_text("")
            continue

        for obj in item["labels"]:

            if "box2d" not in obj:
                continue

            category = obj["category"]

            if category not in CLASS_TO_ID:
                continue

            bbox = obj["box2d"]

            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            class_id = CLASS_TO_ID[category]

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        label_file.write_text("\n".join(lines))

    print(f"[INFO] YOLO labels saved to {output_dir}")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Convert BDD annotations to YOLO format"
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Image directory",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="BDD JSON annotations",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for YOLO labels",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    convert_dataset(
        args.images,
        args.annotations,
        args.output,
    )


if __name__ == "__main__":
    main()