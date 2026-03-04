"""
Enhanced data analysis utilities for BDD100K object detection.

This module computes dataset statistics including:

- Class distribution (log-scale visualization)
- Train/Validation split comparison
- Bounding box statistics (width, height, area, aspect ratio)
- Small object detection
- Object density per image
- Aspect ratio anomaly detection
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


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Class Distribution
# ---------------------------------------------------------

def compute_class_distribution(
    annotations: Dict[str, List[Annotation]],
) -> Dict[str, int]:
    """Compute object instance counts per class."""

    counts = {cls: 0 for cls in DETECTION_CLASSES}

    for anns in annotations.values():
        for ann in anns:
            counts[ann.category] += 1

    return counts


def plot_class_distribution(counts: Dict[str, int], split: str) -> None:
    """Plot class distribution with log scale."""

    classes = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, values)

    # Important for long-tail datasets like BDD
    plt.yscale("log")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of instances (log scale)")
    plt.title(f"BDD100K Class Distribution ({split})")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"class_distribution_{split}.png")
    plt.close()


# ---------------------------------------------------------
# Bounding Box Statistics
# ---------------------------------------------------------

def compute_bbox_statistics(
    annotations: Dict[str, List[Annotation]],
) -> pd.DataFrame:
    """Compute bbox width, height, area and aspect ratio."""

    records = []

    for anns in annotations.values():
        for ann in anns:

            bbox = ann.bbox

            aspect_ratio = 0
            if bbox.height > 0:
                aspect_ratio = bbox.width / bbox.height

            records.append(
                {
                    "category": ann.category,
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": bbox.area,
                    "aspect_ratio": aspect_ratio,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Small Object Detection
# ---------------------------------------------------------

def detect_small_objects(
    annotations: Dict[str, List[Annotation]],
) -> pd.DataFrame:
    """Detect bounding boxes smaller than threshold."""

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


# ---------------------------------------------------------
# Aspect Ratio Anomalies
# ---------------------------------------------------------

def detect_aspect_ratio_anomalies(
    bbox_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect extremely thin or wide bounding boxes.
    """

    anomalies = bbox_df[
        (bbox_df["aspect_ratio"] > 5)
        | (bbox_df["aspect_ratio"] < 0.2)
    ]

    return anomalies


# ---------------------------------------------------------
# Object Density
# ---------------------------------------------------------

def compute_object_density(
    annotations: Dict[str, List[Annotation]],
) -> pd.DataFrame:
    """
    Compute number of objects per image.
    """

    records = []

    for image_id, anns in annotations.items():

        records.append(
            {
                "image_id": image_id,
                "num_objects": len(anns),
            }
        )

    return pd.DataFrame(records)

def compute_bbox_centers(
    annotations: Dict[str, List[Annotation]]
) -> pd.DataFrame:
    """
    Compute the center coordinates of each bounding box.

    This is useful for analyzing spatial bias in the dataset,
    such as cars appearing mostly in the bottom region of images
    and traffic lights appearing near the top.

    Args:
        annotations: Mapping from image_id to annotation list.

    Returns:
        DataFrame containing category and bbox center coordinates.
    """

    records = []

    for image_id, anns in annotations.items():
        for ann in anns:
            bbox = ann.bbox

            center_x = bbox.x1 + bbox.width / 2
            center_y = bbox.y1 + bbox.height / 2

            records.append(
                {
                    "image_id": image_id,
                    "category": ann.category,
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

    return pd.DataFrame(records)


def plot_spatial_heatmap(
    centers_df: pd.DataFrame,
    figures_dir: Path,
    split: str,
) -> None:
    """
    Visualize spatial distribution of object centers.

    This helps identify location bias where objects consistently
    appear in particular regions of the image.

    Args:
        centers_df: DataFrame containing bbox center coordinates.
        figures_dir: Directory to store generated figures.
        split: Dataset split name (train or val).
    """

    plt.figure(figsize=(6, 6))

    plt.hexbin(
        centers_df["center_x"],
        centers_df["center_y"],
        gridsize=60,
    )

    plt.xlabel("Image X Position")
    plt.ylabel("Image Y Position")
    plt.title(f"Object Spatial Distribution ({split})")

    output_path = figures_dir / f"spatial_heatmap_{split}.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_bbox_area_distribution(
    bbox_df: pd.DataFrame,
    figures_dir: Path,
    split: str,
) -> None:
    """
    Plot histogram of bounding box areas.

    This analysis reveals the distribution of object sizes in
    the dataset, which affects detection difficulty.

    Args:
        bbox_df: DataFrame containing bounding box statistics.
        figures_dir: Directory to save output figures.
        split: Dataset split name.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        bbox_df["area"],
        bins=100,
        log=True,
    )

    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Bounding Box Area Distribution ({split})")

    output_path = figures_dir / f"bbox_area_distribution_{split}.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_object_density(density_df: pd.DataFrame, split: str) -> None:
    """Plot histogram of objects per image."""

    plt.figure(figsize=(8, 5))

    plt.hist(density_df["num_objects"], bins=50)

    plt.xlabel("Objects per Image")
    plt.ylabel("Frequency")
    plt.title(f"Object Density Distribution ({split})")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f"object_density_{split}.png")
    plt.close()

def plot_scene_density_per_class(
    annotations: Dict[str, List[Annotation]],
    figures_dir: Path,
    split: str,
) -> None:
    """
    Plot scene density per object class.

    Scene density refers to the number of objects present in
    images containing a particular class. Dense scenes often
    indicate urban environments with heavy traffic.

    Args:
        annotations: Mapping from image_id to annotation list.
        figures_dir: Directory for saving figures.
        split: Dataset split name.
    """

    class_density = {}

    for image_id, anns in annotations.items():
        object_count = len(anns)

        for ann in anns:
            class_density.setdefault(ann.category, []).append(
                object_count
            )

    density_df = pd.DataFrame(
        {cls: pd.Series(values) for cls, values in class_density.items()}
    )

    plt.figure(figsize=(12, 6))

    density_df.boxplot()

    plt.xticks(rotation=45)
    plt.ylabel("Objects per Image")
    plt.title(f"Scene Density per Class ({split})")

    output_path = figures_dir / f"class_scene_density_{split}.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
# ---------------------------------------------------------
# Train/Val Comparison
# ---------------------------------------------------------

def compare_splits(
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
) -> pd.DataFrame:
    """
    Compare class distributions between train and validation splits.
    """

    df = pd.DataFrame(
        {
            "train": train_counts,
            "val": val_counts,
        }
    ).fillna(0)

    df["train_ratio"] = df["train"] / df["train"].sum()
    df["val_ratio"] = df["val"] / df["val"].sum()

    df["difference"] = abs(df["train_ratio"] - df["val_ratio"])

    return df


# ---------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------

def run_analysis(label_file: Path, image_dir: Path, split: str):

    ensure_output_dirs()

    print(f"[INFO] Loading annotations for {split}")

    annotations = load_annotations(label_file)

    dataset = BDDDetectionDataset(image_dir, annotations)

    # -------------------------
    # Class Distribution
    # -------------------------

    class_counts = compute_class_distribution(annotations)

    pd.DataFrame.from_dict(
        class_counts,
        orient="index",
        columns=["count"],
    ).to_csv(TABLES_DIR / f"class_distribution_{split}.csv")

    plot_class_distribution(class_counts, split)

    # -------------------------
    # Bounding Box Statistics
    # -------------------------

    bbox_df = compute_bbox_statistics(annotations)

    bbox_df.to_csv(
        TABLES_DIR / f"bbox_statistics_{split}.csv",
        index=False,
    )

    # -------------------------
    # Aspect Ratio Anomalies
    # -------------------------

    anomalies = detect_aspect_ratio_anomalies(bbox_df)

    anomalies.to_csv(
        TABLES_DIR / f"aspect_ratio_anomalies_{split}.csv",
        index=False,
    )

    # -------------------------
    # Small Objects
    # -------------------------

    small_objects_df = detect_small_objects(annotations)

    small_objects_df.to_csv(
        TABLES_DIR / f"small_objects_{split}.csv",
        index=False,
    )

    # -------------------------
    # Object Density
    # -------------------------

    density_df = compute_object_density(annotations)

    density_df.to_csv(
        TABLES_DIR / f"object_density_{split}.csv",
        index=False,
    )

    plot_object_density(density_df, split)

    # Spatial distribution
    centers_df = compute_bbox_centers(annotations)
    plot_spatial_heatmap(centers_df, FIGURES_DIR, split)

    # Bounding box size distribution
    plot_bbox_area_distribution(bbox_df, FIGURES_DIR, split)

    # Scene density per class
    plot_scene_density_per_class(annotations, FIGURES_DIR, split)

    # -------------------------
    # Console Summary
    # -------------------------

    total_objects = sum(class_counts.values())

    small_ratio = len(small_objects_df) / total_objects

    print("\n------ DATASET SUMMARY ------")
    print(f"Split: {split}")
    print(f"Images: {len(dataset)}")
    print(f"Total Objects: {total_objects}")
    print(f"Small Objects (<{MIN_BBOX_SIZE}px): {len(small_objects_df)}")
    print(f"Small Object Ratio: {small_ratio:.2%}")
    print(f"Aspect Ratio Anomalies: {len(anomalies)}")
    print("-----------------------------\n")
    print(bbox_df["area"].describe())
    print("-----------------------------\n")
    print(bbox_df["aspect_ratio"].describe())
    print("-----------------------------\n")
    print(density_df["num_objects"].describe())

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="BDD100K Object Detection Data Analysis"
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to BDD100K JSON labels",
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
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_analysis(
        label_file=args.labels,
        image_dir=args.images,
        split=args.split,
    )