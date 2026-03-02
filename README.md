# BDD100K Object Detection Analysis

This repository provides an end-to-end, reproducible pipeline for
data analysis, model training, and evaluation on the BDD100K object
detection dataset.

The focus of this project is on understanding the dataset through
systematic analysis, building a baseline object detection model, and
evaluating its performance using both quantitative and qualitative
methods. The entire pipeline is designed to be reproducible using
Docker.

---

## Project Scope

### Dataset
- BDD100K – Object Detection
- Splits used: train and validation
- Labels: JSON annotations with bounding boxes

### Object Detection Classes
- person
- rider
- car
- bus
- truck
- bike
- motor
- traffic light
- traffic sign
- train

### Out of Scope
- Drivable area
- Lane marking
- Semantic segmentation

---


## Repository Structure

```text
bdd100k-object-detection-analysis/
├── src/
│   ├── config.py        # Global configuration
│   ├── parser.py        # BDD100K JSON parser
│   ├── dataset.py       # Dataset abstractions
│   ├── analysis.py      # Data analysis pipeline
│   ├── visualize.py    # Qualitative visualization
│   ├── dashboard.py    # Streamlit dashboard
│   ├── train.py        # Model training (subset supported)
│   └── evaluate.py     # Model evaluation
├── outputs/
│   ├── figures/         # Generated plots
│   ├── tables/          # CSV statistics
│   └── samples/         # Qualitative samples
├── Dockerfile
├── requirements.txt
├── README.md
└── REPORT.md
```

## Data Analysis

The data analysis stage includes:
- Class distribution analysis
- Train vs validation split analysis
- Bounding box statistics (width, height, area)
- Detection of anomalous samples (e.g., very small objects)
- Identification of interesting or unique samples
- Visualization of dataset statistics

All analysis outputs are saved under the outputs/ directory.

### Run Data Analysis
'''
python src/analysis.py \
  --labels data/labels/bdd100k_labels_images_train.json \
  --images data/images/train \
  --split train
'''
Repeat the command for validation by changing the paths and using
--split val.

---

## Dashboard

An interactive Streamlit dashboard is provided to visualize the
analysis results.

Run the dashboard using:

streamlit run src/dashboard.py

---

## Model Training

A Faster R-CNN (ResNet-50 + FPN) model from torchvision is used as
a baseline object detector.

### Why Faster R-CNN
- Strong and well-understood two-stage detector
- Performs well on complex driving scenes
- Suitable as a baseline for small and rare objects

### Training Features
- Custom PyTorch Dataset for BDD100K
- Support for subset training
- Training for a small number of epochs

### Run Training (Example)
```text
python src/train.py \
  --labels data/labels/bdd100k_labels_images_train.json \
  --images data/images/train \
  --epochs 1 \
  --subset 0.02
```
The trained model is saved to outputs/model.pth.

---

## Evaluation and Visualization

### Quantitative Evaluation
- Per-class True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Precision and Recall (IoU ≥ 0.5)

Metrics are saved to:
outputs/tables/evaluation_metrics.csv

### Qualitative Evaluation
- Visualization of failure cases
- Identification of missed detections

### Run Evaluation

python src/evaluate.py \
  --labels data/labels/bdd100k_labels_images_val.json \
  --images data/images/val \
  --weights outputs/model.pth

---

## Docker Usage

This project is fully containerized and can be run without any
additional installations.

### Build Docker Image

docker build -t bdd100k-analysis .

### Run Container (mount dataset)

docker run -it \
  -v /path/to/bdd100k:/app/data \
  -p 8501:8501 \
  bdd100k-analysis

---

## Dataset Setup

The BDD100K dataset is not included in this repository due to size
constraints.

Expected directory structure:

```text
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── bdd100k_labels_images_train.json
│   └── bdd100k_labels_images_val.json
```

## Coding Standards

- Python 3.10
- PEP8 compliant
- Type hints and docstrings
- Script-driven (no notebooks required)
- Modular and reproducible design

---

## Report

Detailed analysis, observations, and findings are documented in
REPORT.md.

---

## Summary

This repository demonstrates:
- Structured dataset analysis
- Clean ML engineering practices
- A complete object detection pipeline
- Reproducibility via Docker
