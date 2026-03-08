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
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DETECTION_CLASSES,
    FIGURES_DIR,
    TABLES_DIR,
    MIN_BBOX_SIZE,
    IMAGE_DIR_TRAIN,
    IMAGE_DIR_VAL,
    LABEL_FILE_TRAIN,
    LABEL_FILE_VAL,
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
    """Plot class distribution sorted from largest to smallest."""

    # Sort classes by count (descending)
    sorted_items = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    classes = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 5))

    plt.bar(classes, values)

    # Important for long-tail datasets
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


def plot_train_val_class_distribution(train_counts, val_counts):
    """
    Plot train vs validation class distribution in one figure.
    """

    # Convert to DataFrame
    df = pd.DataFrame({
        "train": train_counts,
        "val": val_counts,
    })

    # Sort by train counts (descending) to show long tail
    df = df.sort_values("train", ascending=False)

    classes = df.index.tolist()
    train_values = df["train"].values
    val_values = df["val"].values

    x = range(len(classes))
    width = 0.4

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        train_values,
        width=width,
        label="Train"
    )

    plt.bar(
        [i + width / 2 for i in x],
        val_values,
        width=width,
        label="Validation"
    )

    plt.yscale("log")

    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("Number of instances (log scale)")
    plt.title("BDD100K Class Distribution (Train vs Validation)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_train_val.png")
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

def plot_coco_area_distribution(
    bbox_df: pd.DataFrame, 
    figures_dir: Path, 
    split: str
) -> None:
    """
    Categorizes bounding boxes by COCO area standards and plots the distribution.
    """
    # Define thresholds
    small_thresh = 32**2
    large_thresh = 96**2

    # Categorize areas
    def categorize(area):
        if area < small_thresh:
            return "Small"
        elif area <= large_thresh:
            return "Medium"
        else:
            return "Large"

    bbox_df["scale_category"] = bbox_df["area"].apply(categorize)
    
    # Calculate counts and percentages
    counts = bbox_df["scale_category"].value_counts().reindex(["Small", "Medium", "Large"])
    percentages = (counts / len(bbox_df)) * 100

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    bars = plt.bar(counts.index, counts.values, color=['#ff9999','#66b3ff','#99ff99'], edgecolor='black')

    # Add labels on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            yval + (max(counts.values) * 0.01), 
            f'{int(yval)}\n({percentages.iloc[i]:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontweight='bold'
        )

    plt.xlabel("Object Scale (COCO Standards)")
    plt.ylabel("Number of Instances")
    plt.title(f"BDD100K Object Scale Distribution ({split})\nSmall < $32^2$, Medium $32^2$-$96^2$, Large > $96^2$")
    
  
    plt.yscale("log")

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = figures_dir / f"bbox_coco_scale_{split}.png"
    plt.savefig(output_path)
    plt.close()


def plot_aspect_ratio_clusters(bbox_df: pd.DataFrame, figures_dir: Path, split: str) -> None:
    """
    Fast visualization of width vs height distribution using hexbin.
    Much faster than seaborn KDE for large datasets.
    """

    plt.figure(figsize=(10, 10))

    # Remove extreme outliers
    q_w = bbox_df["width"].quantile(0.99)
    q_h = bbox_df["height"].quantile(0.99)

    filtered_df = bbox_df[
        (bbox_df["width"] < q_w) &
        (bbox_df["height"] < q_h)
    ]

    x = filtered_df["width"].values
    y = filtered_df["height"].values

    plt.hexbin(
        x,
        y,
        gridsize=60,
        bins="log",
        cmap="inferno"
    )

    plt.colorbar(label="Log Density")

    # Add aspect ratio reference lines
    x_vals = np.array([0, q_w])

    plt.plot(x_vals, x_vals * 1, '--', color='#00FFFF', linewidth=2.5, label="1:1")
    plt.plot(x_vals, x_vals * 2, ':', color='#00FF00', linewidth=2.5, label="1:2")
    plt.plot(x_vals, x_vals * 0.5, '-.', color='#FF00FF', linewidth=2.5, label="2:1")

    plt.xlabel("Bounding Box Width")
    plt.ylabel("Bounding Box Height")

    plt.title(f"Width vs Height Density (Anchor Box Insights) - {split}")

    plt.legend()

    plt.tight_layout()

    output_path = figures_dir / f"aspect_ratio_density_{split}.png"
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

def plot_scene_complexity(density_df: pd.DataFrame, figures_dir: Path, split: str) -> None:
    """
    Categorizes images by number of objects to show scene complexity.
    Optimized version without pandas.cut.
    """

    counts = density_df["num_objects"].values

    sparse = (counts <= 5).sum()
    moderate = ((counts > 5) & (counts <= 15)).sum()
    dense = ((counts > 15) & (counts <= 30)).sum()
    crowded = (counts > 30).sum()

    labels = ['Sparse (0-5)', 'Moderate (6-15)', 'Dense (16-30)', 'Crowded (30+)']
    values = [sparse, moderate, dense, crowded]

    plt.figure(figsize=(10, 6))

    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']

    bars = plt.bar(labels, values, color=colors, edgecolor='black')

    total = len(counts)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 5,
            f'{(height/total)*100:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.title(f"Scene Complexity Distribution ({split})")
    plt.xlabel("Objects per Image")
    plt.ylabel("Number of Images")
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / f"scene_complexity_{split}.png")
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

    total_objects = sum(class_counts.values())

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

    bbox_stats = bbox_df[["width", "height", "area", "aspect_ratio"]].describe()
    bbox_stats.to_csv(
    TABLES_DIR / f"bbox_statistics_summary_{split}.csv"
    )
    # -------------------------
    # Aspect Ratio Anomalies
    # -------------------------

    anomalies = detect_aspect_ratio_anomalies(bbox_df)

    anomaly_stats = pd.DataFrame({
    "metric": ["total_boxes", "aspect_ratio_anomalies"],
    "value": [len(bbox_df), len(anomalies)]
    })

    anomaly_stats.to_csv(
        TABLES_DIR / f"aspect_ratio_anomalies_summary_{split}.csv",
        index=False
    )

    # -------------------------
    # Small Objects
    # -------------------------

    small_objects_df = detect_small_objects(annotations)

    small_stats = pd.DataFrame({
    "metric": ["total_objects", "small_objects", "small_object_ratio"],
    "value": [
        total_objects,
        len(small_objects_df),
        len(small_objects_df) / total_objects
        ]
    })

    small_stats.to_csv(
        TABLES_DIR / f"small_object_summary_{split}.csv",
        index=False
    )

    # -------------------------
    # Object Density
    # -------------------------

    density_df = compute_object_density(annotations)

    density_stats = density_df["num_objects"].describe()

    density_stats.to_csv(
        TABLES_DIR / f"object_density_summary_{split}.csv"
    )
    plot_object_density(density_df, split)

    # Spatial distribution
    centers_df = compute_bbox_centers(annotations)
    plot_spatial_heatmap(centers_df, FIGURES_DIR, split)
    plot_bbox_area_distribution(bbox_df, FIGURES_DIR, split)
    plot_coco_area_distribution(bbox_df, FIGURES_DIR, split)
    plot_aspect_ratio_clusters(bbox_df, FIGURES_DIR, split)
    plot_scene_density_per_class(annotations, FIGURES_DIR, split)
    plot_scene_complexity(density_df, FIGURES_DIR, split)

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
    return class_counts

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------




if __name__ == "__main__":

    ensure_output_dirs()

    # -------------------------
    # Run detailed analysis
    # -------------------------

    train_counts=run_analysis(
        label_file=LABEL_FILE_TRAIN,
        image_dir=IMAGE_DIR_TRAIN,
        split="train",
    )

    val_counts=run_analysis(
        label_file=LABEL_FILE_VAL,
        image_dir=IMAGE_DIR_VAL,
        split="val",
    )

    # -------------------------
    # Train vs Validation plot
    # -------------------------
    plot_train_val_class_distribution(train_counts, val_counts)