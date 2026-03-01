# BDD100K Object Detection – Data Analysis, Modeling, and Evaluation Report

## 1. Introduction

This report documents the analysis, modeling, and evaluation performed on the
BDD100K object detection dataset. The objective of this work is to understand
the dataset characteristics, identify potential challenges, build a baseline
object detection model, and evaluate its performance using both quantitative
and qualitative methods.

The analysis is structured to emphasize **data understanding before modeling**,
followed by a reproducible training and evaluation pipeline.

---

## 2. Dataset Overview

The BDD100K dataset is a large-scale autonomous driving dataset containing
images captured under diverse environmental conditions such as varying
weather, lighting, and traffic density.

### Dataset Scope
- Task: Object Detection
- Splits used: Train and Validation
- Total detection classes: 10
- Annotation format: Bounding boxes in JSON

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

Drivable area segmentation, lane markings, and other semantic segmentation
annotations were explicitly excluded from this analysis.

---

## 3. Data Analysis

### 3.1 Class Distribution Analysis

The distribution of object instances across classes reveals a **strong class
imbalance**. The `car` class dominates the dataset, while classes such as
`train`, `motor`, and `bus` appear far less frequently.

Key observations:
- The dataset exhibits a long-tail distribution.
- Rare classes have significantly fewer samples compared to common classes.
- This imbalance is expected to impact model recall for rare categories.

Such imbalance highlights the importance of per-class evaluation metrics rather
than relying solely on aggregate performance.

---

### 3.2 Train vs Validation Split Analysis

The class distribution between the training and validation splits was compared
to assess consistency.

Findings:
- Overall class distribution trends are similar between splits.
- Rare classes are underrepresented in the validation set.
- Evaluation metrics for rare classes are therefore more sensitive to noise.

This suggests that validation performance for rare classes should be interpreted
with caution.

---

### 3.3 Bounding Box Statistics

Bounding box statistics were analyzed in terms of width, height, and area.

Notable patterns:
- Traffic lights and traffic signs typically have small bounding box areas.
- Vehicles such as cars, buses, and trucks occupy larger bounding box regions.
- Aspect ratios vary significantly across classes.

The prevalence of small objects (especially traffic signs and lights) indicates
that small-object detection is a key challenge in this dataset.

---

### 3.4 Image-Level Statistics

The number of objects per image varies widely:
- Many images contain only a few objects.
- Urban scenes can contain a large number of annotated objects.

Crowded scenes introduce challenges related to occlusion and overlapping
objects, which can negatively affect detection performance.

---

### 3.5 Anomalies and Edge Cases

Several anomalies and challenging cases were identified:
- Extremely small bounding boxes (less than 10 pixels in width or height).
- Images with a very high object count.
- Rare classes appearing in limited visual contexts.

These cases were flagged and visualized to better understand their impact on
model performance.

---

## 4. Qualitative Data Analysis

Qualitative visualization was used to inspect:
- Smallest and largest object instances per class
- Highly crowded scenes
- Rare-class examples

These visualizations confirmed that:
- Small objects are often visually ambiguous.
- Rare classes are frequently underrepresented and context-specific.
- Crowded scenes introduce significant occlusion.

This qualitative analysis complements the quantitative statistics and helps
interpret model failures.

---

## 5. Model Selection and Training

### 5.1 Model Choice

A **Faster R-CNN with a ResNet-50 FPN backbone** was selected as the baseline
object detection model.

Reasons for this choice:
- Well-established two-stage detector
- Strong performance on complex scenes
- Suitable baseline for datasets with small and rare objects
- Widely used and easy to interpret

---

### 5.2 Training Setup

Due to computational constraints, the model was trained:
- Using pretrained weights
- On a subset of the training data
- For a limited number of epochs (1 epoch)

A custom PyTorch dataset loader was implemented to load BDD100K annotations
and images into the training pipeline.

The objective of training was to demonstrate a **working and reproducible
training pipeline**, rather than to achieve optimal accuracy.

---

## 6. Evaluation Metrics

### 6.1 Quantitative Metrics

The following metrics were computed on the validation dataset:
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall

An IoU threshold of 0.5 was used to determine correct detections.

Precision and recall were chosen because:
- They provide insight into class-wise performance
- They highlight the impact of class imbalance
- They expose failure modes such as missed detections

---

### 6.2 Quantitative Results Analysis

The evaluation results indicate:
- Higher precision and recall for frequent classes such as `car`.
- Lower recall for rare and small-object classes such as `traffic sign`,
  `traffic light`, and `train`.

These results align with the observations from the data analysis stage.

---

## 7. Qualitative Evaluation and Failure Analysis

Qualitative evaluation focused on inspecting failure cases, particularly:
- Images where ground-truth objects were missed entirely
- Scenes with small or heavily occluded objects
- Crowded urban environments

Observed failure patterns:
- Small objects are frequently missed.
- Rare classes suffer from low recall.
- Crowded scenes increase false negatives.

These failure modes are consistent with the dataset characteristics identified
during data analysis.

---

## 8. Connecting Data Analysis to Model Performance

The evaluation results strongly correlate with the data analysis findings:
- Class imbalance leads to uneven performance across categories.
- Small bounding boxes result in lower detection recall.
- Rare classes are insufficiently represented for robust learning.

This highlights the importance of thorough dataset analysis prior to model
development.

---

## 9. Suggested Improvements

Based on the analysis and evaluation, potential improvements include:
- Class-aware sampling or reweighting strategies
- Data augmentation targeted at small objects
- Higher-resolution input images
- Collecting additional data for rare classes
- Using loss functions designed for class imbalance (e.g., focal loss)

These improvements could help mitigate the identified weaknesses.

---

## 10. Conclusion

This project demonstrates a complete and reproducible object detection pipeline
for the BDD100K dataset, covering:
- Structured data analysis
- Identification of dataset challenges
- Baseline model training
- Quantitative and qualitative evaluation
- Data-driven performance interpretation

The results emphasize that dataset characteristics such as class imbalance,
small objects, and crowded scenes play a critical role in determining object
detection performance.

This analysis provides a strong foundation for future model improvements and
more advanced experimentation.