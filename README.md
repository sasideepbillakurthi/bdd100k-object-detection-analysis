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
│   │
│   ├── models/
│   │   ├── faster_rcnn.py
│   │   ├── swin_faster_rcnn.py
│   ├── config.py        # Global configuration
│   ├── parser.py        # BDD100K JSON parser
│   ├── dataset.py       # Dataset abstractions
│   ├── analysis.py      # Data analysis pipeline
│   ├── train.py        # Model training (subset supported)
│   └── evaluate.py     # Model evaluation
│   └── qualitative_eval.py     # Qualitative evaluation
|
├── outputs/
│   ├── figures/         # Generated plots
│   ├── tables/          # CSV statistics
│   └── qualitative/         # Qualitative samples
├── Dockerfile
├── requirements.txt
├── README.md
└── REPORT.md
```


## Dataset Analysis

The dataset was analyzed to understand important characteristics that influence detection performance.

Key analyses performed:

- **Class distribution**
- **Bounding box size distribution**
- **Aspect ratio distribution**
- **Scene complexity (objects per image)**
- **Small object prevalence**
- **Spatial object distribution**

Generated analysis outputs are stored in:


All analysis outputs are saved under the outputs/ directory.

### Run Data Analysis
```text
python -m src.analysis
```
---

## Model Architecture

The detection model used in this project is:

**Faster R-CNN with a Swin Transformer backbone and Feature Pyramid Network (FPN)**.

Architecture components:

1. **Swin Transformer Backbone**
2. **Feature Pyramid Network (FPN)**
3. **Region Proposal Network (RPN)**
4. **ROI-based Detection Head**

This architecture was selected because the dataset analysis revealed:

- many **small objects**
- significant **scale variation**
- **crowded urban scenes**

FPN improves small-object detection, while the Swin Transformer backbone captures richer contextual features.

---

## Training

The training pipeline supports both:

- **Faster R-CNN (ResNet backbone)**
- **Swin Transformer + Faster R-CNN**

Example training command:
```text
python -m src.train --model swin --epochs 1 --subset 0.5
```


---

## Evaluation

Model performance was evaluated on the **BDD100K validation dataset**.

### Quantitative Metrics

The following metrics were used:

| Metric | Value |
|------|------|
| mAP@0.5 | 0.4099 |
| mIoU | 0.7271 |

Additional evaluation includes:

- per-class precision
- per-class recall
- confusion matrix
- precision–recall curves

---

### Qualitative Analysis

Model predictions were visualized by overlaying predicted bounding boxes with ground-truth annotations.

Visualization conventions:

- **Green boxes → Ground Truth**
- **Red boxes → Model Predictions**



### Run Evaluation
```text
python -m src.evaluate --model swin --weights outputs/models/model.pth
```
---

## Docker Usage

This project is fully containerized and can be run without any
additional installations.

### Build Docker Image

docker build -t bdd100k-analysis .

### Run Container (mount dataset)

docker run -it --rm \
--gpus all \
-v "$(pwd)/data:/workspace/bdd100k-object-detection-analysis/data" \
bdd100k-analysis

---


---

## Results

The model demonstrates:

**Strengths**

- accurate localization
- strong detection for common objects such as cars
- reasonable performance in diverse environments

**Challenges**

- small object detection
- crowded scenes
- class confusion between visually similar objects

These behaviors are consistent with the dataset characteristics identified during analysis.

---

# Report

A detailed explanation of the methodology, experiments, and findings is provided in: [Report](REPORT.md).


