# src/config.py
"""
Global configuration for BDD100K object detection analysis.
"""

from pathlib import Path

# -----------------------------
# Dataset configuration
# -----------------------------
DATA_DIR = Path("data")

IMAGE_DIR_TRAIN = DATA_DIR/ "bdd100k_images_100k" / "bdd100k" / "images" / "train"
IMAGE_DIR_VAL = DATA_DIR / "bdd100k_images_100k" / "bdd100k" / "images" / "val"

LABEL_FILE_TRAIN = DATA_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
LABEL_FILE_VAL = DATA_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

# -----------------------------
# Output configuration
# -----------------------------
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
SAMPLES_DIR = OUTPUT_DIR / "samples"

# -----------------------------
# Object detection classes
# -----------------------------
DETECTION_CLASSES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

# -----------------------------
# Analysis thresholds
# -----------------------------
MIN_BBOX_SIZE = 10  # pixels
MAX_OBJECTS_PER_IMAGE = 50