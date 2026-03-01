"""
Data analysis utilities for BDD100K object detection.

This module computes dataset statistics such as class distribution,
bounding box properties, train/validation split behavior, and
identifies anomalous or interesting samples.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DETECTION_CLASSES,
    FIGURES_DIR,
    MIN_BBOX_SIZE,
    TABLES_DIR,
)
from src.dataset import BDDDetectionDataset
from src.parser import Annotation, load_annotations


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_class_distribution(
    annotations: Dict[str, List[Annotation]],
) -> Dict[str, int]:
    """
    Compute total number of object instances per class.

    Args:
        annotations (Dict[str, List[Annotation]]): Parsed annotations.

    Returns:
        Dict[str, int]: Mapping from class name to instance count.
    """
    counts = {cls: 0 for cls in DETECTION_CLASSES}

    for anns in annotations.values():
        for ann in anns:
            counts[ann.category] += 1

    return counts


def compute_bbox_statistics(
    annotations: Dict[str, List[Annotation]],
) -> pd.DataFrame:
    """
    Compute bounding box statistics for all annotations.

    Args:
        annotations (Dict[str, List[Annotation]]): Parsed annotations.

    Returns:
        pd.DataFrame: DataFrame containing bbox width, height, area, class.
    """
    records = []

    for anns in annotations.values():
        for ann in anns:
            bbox = ann.bbox
            records.append(
                {
                    "category": ann.category,
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": bbox.area,
                }
            )

    return pd.DataFrame(records)


def detect_small_objects(
    annotations: Dict[str, List[Annotation]],
) -> pd.DataFrame:
    """
    Identify extremely small bounding boxes.

    Args:
        annotations (Dict[str, List[Annotation]]): Parsed annotations.

    Returns:
        pd.DataFrame: DataFrame of small-object annotations.
    """
    records = []

    for image_id, anns in annotations.items():
        for ann in anns:
            if (
                ann.bbox.width < MIN_BBOX_SIZE
                or ann.bbox.height < MIN_BBOX_SIZE
            ):
                records.append(
                    {
                        "image_id": image_id,
                        "category": ann.category,
                        "width": ann.bbox.width,
                        "height": ann.bbox.height,
                    }
                )

    return pd.DataFrame(records)


def plot_class_distribution(counts: Dict[str, int], split: str) -> None:
    """
    Plot and save class distribution bar chart.

    Args:
        counts (Dict[str, int]): Class counts.
        split (str): Dataset split name (train or val).
    """
    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of instances")
    plt.title(f"Class Distribution ({split})")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"class_distribution_{split}.png"
    plt.savefig(output_path)
    plt.close()


def run_analysis(label_file: Path, image_dir: Path, split: str) -> None:
    """
    Run full data analysis pipeline for a dataset split.

    Args:
        label_file (Path): Path to label JSON.
        image_dir (Path): Path to image directory.
        split (str): Name of dataset split.
    """
    ensure_output_dirs()

    annotations = load_annotations(label_file)
    dataset = BDDDetectionDataset(image_dir, annotations)

    # ---- Class distribution ----
    class_counts = compute_class_distribution(annotations)
    plot_class_distribution(class_counts, split)

    pd.DataFrame.from_dict(
        class_counts, orient="index", columns=["count"]
    ).to_csv(TABLES_DIR / f"class_distribution_{split}.csv")

    # ---- Bounding box statistics ----
    bbox_df = compute_bbox_statistics(annotations)
    bbox_df.to_csv(TABLES_DIR / f"bbox_statistics_{split}.csv", index=False)

    # ---- Small object detection ----
    small_objects_df = detect_small_objects(annotations)
    small_objects_df.to_csv(
        TABLES_DIR / f"small_objects_{split}.csv", index=False
    )

    print(f"[INFO] Analysis completed for split: {split}")
    print(f"[INFO] Number of images: {len(dataset)}")
    print(f"[INFO] Total objects: {sum(class_counts.values())}")
    print(
        f"[INFO] Small objects (<{MIN_BBOX_SIZE}px): "
        f"{len(small_objects_df)}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BDD100K Object Detection Data Analysis"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to BDD100K label JSON file",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        label_file=args.labels,
        image_dir=args.images,
        split=args.split,
    )