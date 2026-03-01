"""
Dataset abstractions for BDD100K object detection.

This module provides lightweight dataset utilities built on top of the
parsed BDD100K annotations, enabling image-level access, statistics,
and iteration for analysis or training pipelines.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2

from src.config import DETECTION_CLASSES
from src.parser import Annotation


class BDDDetectionDataset:
    """
    Image-centric dataset abstraction for BDD100K object detection.
    """

    def __init__(
        self,
        image_dir: Path,
        annotations: Dict[str, List[Annotation]],
    ) -> None:
        """
        Initialize the dataset.

        Args:
            image_dir (Path): Directory containing images.
            annotations (Dict[str, List[Annotation]]):
                Mapping from image ID to object annotations.
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.image_ids = sorted(annotations.keys())

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        """
        return len(self.image_ids)

    def __iter__(self) -> Iterator[Tuple[str, List[Annotation]]]:
        """
        Iterate over dataset samples.

        Yields:
            Tuple[str, List[Annotation]]:
                Image ID and its corresponding annotations.
        """
        for image_id in self.image_ids:
            yield image_id, self.annotations[image_id]

    def get_annotations(self, image_id: str) -> List[Annotation]:
        """
        Get annotations for a specific image.

        Args:
            image_id (str): Image filename.

        Returns:
            List[Annotation]: List of annotations for the image.
        """
        return self.annotations.get(image_id, [])

    def load_image(self, image_id: str):
        """
        Load an image from disk.

        Args:
            image_id (str): Image filename.

        Returns:
            ndarray: Loaded image in BGR format.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = self.image_dir / image_id

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def get_image_shape(self, image_id: str) -> Tuple[int, int]:
        """
        Get image height and width.

        Args:
            image_id (str): Image filename.

        Returns:
            Tuple[int, int]: (height, width)
        """
        image = self.load_image(image_id)
        height, width = image.shape[:2]
        return height, width

    def class_counts(self) -> Dict[str, int]:
        """
        Count total object instances per class.

        Returns:
            Dict[str, int]: Mapping from class name to instance count.
        """
        counts = {cls: 0 for cls in DETECTION_CLASSES}

        for annotations in self.annotations.values():
            for ann in annotations:
                counts[ann.category] += 1

        return counts

    def objects_per_image(self) -> List[int]:
        """
        Get the number of objects in each image.

        Returns:
            List[int]: Object counts per image.
        """
        return [len(anns) for anns in self.annotations.values()]

    def images_with_class(self, category: str) -> List[str]:
        """
        Get image IDs that contain a given object class.

        Args:
            category (str): Object category.

        Returns:
            List[str]: Image IDs containing the category.
        """
        if category not in DETECTION_CLASSES:
            raise ValueError(f"Unknown category: {category}")

        return [
            image_id
            for image_id, anns in self.annotations.items()
            if any(ann.category == category for ann in anns)
        ]