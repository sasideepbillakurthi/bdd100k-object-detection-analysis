"""
Visualization utilities for BDD100K object detection analysis.

This module provides helper functions to visualize statistics and
qualitative samples such as bounding boxes, crowded scenes, and
interesting edge cases.
"""

from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, SAMPLES_DIR
from src.dataset import BDDDetectionDataset
from src.parser import Annotation


def ensure_visual_dirs() -> None:
    """Create visualization output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def plot_bbox_area_distribution(
    bbox_df: pd.DataFrame, split: str
) -> None:
    """
    Plot bounding box area distribution per class.

    Args:
        bbox_df (pd.DataFrame): Bounding box statistics dataframe.
        split (str): Dataset split name.
    """
    plt.figure(figsize=(10, 5))

    for category in bbox_df["category"].unique():
        areas = bbox_df[bbox_df["category"] == category]["area"]
        plt.hist(areas, bins=50, alpha=0.5, label=category)

    plt.xlabel("Bounding Box Area (pixels)")
    plt.ylabel("Frequency")
    plt.title(f"Bounding Box Area Distribution ({split})")
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"bbox_area_distribution_{split}.png"
    plt.savefig(output_path)
    plt.close()


def draw_bounding_boxes(
    image,
    annotations: List[Annotation],
    color=(0, 255, 0),
) :
    """
    Draw bounding boxes on an image.

    Args:
        image: OpenCV image.
        annotations (List[Annotation]): List of annotations.
        color (tuple): BGR color for boxes.

    Returns:
        Image with drawn bounding boxes.
    """
    img = image.copy()

    for ann in annotations:
        bbox = ann.bbox
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            ann.category,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


def save_crowded_samples(
    dataset: BDDDetectionDataset,
    min_objects: int = 15,
    max_samples: int = 5,
) -> None:
    """
    Save images with a large number of objects (crowded scenes).

    Args:
        dataset (BDDDetectionDataset): Dataset instance.
        min_objects (int): Minimum number of objects to consider crowded.
        max_samples (int): Maximum number of images to save.
    """
    ensure_visual_dirs()

    saved = 0
    for image_id, anns in dataset:
        if len(anns) >= min_objects:
            image = dataset.load_image(image_id)
            vis = draw_bounding_boxes(image, anns)

            output_path = SAMPLES_DIR / f"crowded_{image_id}"
            cv2.imwrite(str(output_path), vis)

            saved += 1
            if saved >= max_samples:
                break


def save_extreme_object_samples(
    dataset: BDDDetectionDataset,
    category: str,
) -> None:
    """
    Save smallest and largest object instances for a given class.

    Args:
        dataset (BDDDetectionDataset): Dataset instance.
        category (str): Object category.
    """
    ensure_visual_dirs()

    smallest = None
    largest = None

    for image_id, anns in dataset:
        for ann in anns:
            if ann.category != category:
                continue

            area = ann.bbox.area
            if smallest is None or area < smallest[0]:
                smallest = (area, image_id, ann)

            if largest is None or area > largest[0]:
                largest = (area, image_id, ann)

    for label, sample in [("smallest", smallest), ("largest", largest)]:
        if sample is None:
            continue

        _, image_id, ann = sample
        image = dataset.load_image(image_id)
        vis = draw_bounding_boxes(image, [ann], color=(0, 0, 255))

        output_path = SAMPLES_DIR / f"{category}_{label}_{image_id}"
        cv2.imwrite(str(output_path), vis)