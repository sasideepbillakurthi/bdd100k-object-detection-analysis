"""
Parser utilities for BDD100K object detection annotations.

This module provides data structures and functions to load and parse
BDD100K detection JSON files into clean Python objects for analysis
and modeling.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.config import DETECTION_CLASSES


@dataclass
class BoundingBox:
    """
    Represents a 2D bounding box in image pixel coordinates.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return bounding box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return bounding box height."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Return bounding box area."""
        return self.width * self.height


@dataclass
class Annotation:
    """
    Represents a single object annotation in an image.
    """

    image_id: str
    category: str
    bbox: BoundingBox


def _parse_bbox(box2d: Dict[str, float]) -> Optional[BoundingBox]:
    """
    Parse a BDD100K 'box2d' dictionary into a BoundingBox object.

    Args:
        box2d (Dict[str, float]): Dictionary containing x1, y1, x2, y2.

    Returns:
        Optional[BoundingBox]: Parsed bounding box or None if invalid.
    """
    try:
        return BoundingBox(
            x1=float(box2d["x1"]),
            y1=float(box2d["y1"]),
            x2=float(box2d["x2"]),
            y2=float(box2d["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_annotations(label_file: Path) -> Dict[str, List[Annotation]]:
    """
    Load BDD100K object detection annotations from a JSON file.

    Only annotations belonging to the predefined detection classes
    are considered. Images without valid bounding boxes are skipped.

    Args:
        label_file (Path): Path to BDD100K detection JSON file.

    Returns:
        Dict[str, List[Annotation]]:
            Mapping from image ID to list of object annotations.

    Raises:
        FileNotFoundError: If the label file does not exist.
        ValueError: If the JSON file is malformed.
    """
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    try:
        with label_file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {label_file}") from exc

    annotations: Dict[str, List[Annotation]] = {}

    for item in raw_data:
        image_id = item.get("name")
        labels = item.get("labels", [])

        if not image_id:
            continue

        image_annotations: List[Annotation] = []

        for label in labels:
            category = label.get("category")

            if category not in DETECTION_CLASSES:
                continue

            box2d = label.get("box2d")
            if box2d is None:
                continue

            bbox = _parse_bbox(box2d)
            if bbox is None:
                continue

            image_annotations.append(
                Annotation(
                    image_id=image_id,
                    category=category,
                    bbox=bbox,
                )
            )

        if image_annotations:
            annotations[image_id] = image_annotations

    return annotations