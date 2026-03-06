"""
YOLOv8 model utilities for BDD100K object detection.

This module provides helper functions for loading,
training, and running inference using YOLOv8 from
the Ultralytics framework.
"""

from pathlib import Path
from ultralytics import YOLO


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """
    Load a YOLOv8 model.

    Parameters
    ----------
    weights : str
        Path to pretrained YOLO weights.

    Returns
    -------
    YOLO
        Ultralytics YOLO model instance.
    """

    model = YOLO(weights)

    return model


def train_model(
    data_yaml: str,
    epochs: int = 1,
    batch_size: int = 8,
    image_size: int = 640,
    weights: str = "best.pt",
    project_dir: str = "outputs/yolo",
) -> None:
    """
    Train YOLOv8 model on BDD dataset.

    Parameters
    ----------
    data_yaml : str
        Path to dataset YAML file.

    epochs : int
        Number of training epochs.

    batch_size : int
        Training batch size.

    image_size : int
        Input image resolution.

    weights : str
        Initial pretrained weights.

    project_dir : str
        Directory to save outputs.
    """

    model = load_model(weights)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=project_dir,
    )

    print("[INFO] YOLOv8 training finished.")


def run_inference(
    model_path: str,
    image_path: str,
    output_dir: str = "outputs/predictions",
):
    """
    Run YOLOv8 inference on an image.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO model.

    image_path : str
        Path to image.

    output_dir : str
        Directory to save predictions.
    """

    model = YOLO(model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=image_path,
        save=True,
        project=output_dir,
    )

    return results