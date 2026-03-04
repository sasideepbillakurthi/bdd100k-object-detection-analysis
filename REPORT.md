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

To understand the distribution of object categories in the dataset, the number of instances for each class was computed using the training split of the dataset.

The training split contains 69,863 images with a total of 1,286,871 annotated object instances across the ten detection classes.

The distribution of object instances per class is shown in Table 1 and visualized in Figure 1.

Table 1: Object Instance Distribution (Train Split)
| Class         | Count   | Percentage |
| ------------- | ------- | ---------- |
| car           | 713,211 | 55.42%     |
| traffic sign  | 239,686 | 18.62%     |
| traffic light | 186,117 | 14.46%     |
| person        | 91,349  | 7.10%      |
| truck         | 29,971  | 2.33%      |
| bus           | 11,672  | 0.91%      |
| bike          | 7,210   | 0.56%      |
| rider         | 4,517   | 0.35%      |
| motor         | 3,002   | 0.23%      |
| train         | 136     | 0.01%      |
---
Observations
Figure 1 shows the class distribution using a logarithmic scale to better visualize rare classes.


The dataset exhibits a strong class imbalance, with a small number of classes dominating the dataset.

The car class alone accounts for 55.42% of all annotated objects, meaning that more than half of the dataset consists of cars. This reflects the natural distribution of objects in driving scenes where vehicles are the most common objects encountered by autonomous systems.

Traffic infrastructure is also highly represented. Traffic signs and traffic lights together account for 33.1% of all object instances, highlighting the importance of road infrastructure elements in autonomous driving perception tasks.

Overall, the dataset follows a long-tail distribution. The three most frequent classes — car, traffic sign, and traffic light — together represent 88.5% of all annotated objects.

In contrast, several classes are extremely underrepresented. For example, the train class appears only 136 times, representing approximately 0.01% of the dataset. This means that cars appear more than 5,200 times more frequently than trains.

Implications for Object Detection

Such class imbalance can significantly influence the behavior of object detection models. Models trained on this dataset may become biased toward frequently occurring classes such as cars, while performance on rare classes like trains, riders, or motorcycles may be limited due to the small number of training examples.

To address this issue, techniques such as class-balanced sampling, focal loss, or data augmentation for rare classes are commonly used when training object detection models on long-tail datasets.

4. Bounding Box Size Distribution

To understand the scale of objects in the dataset, bounding box areas were analyzed for all annotated objects in the training split. Bounding box area is computed as the product of bounding box width and height.

The dataset contains 1,286,871 bounding boxes. Summary statistics of the bounding box areas are shown in Table 2, and the distribution is visualized in Figure 2.

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
4. Bounding Box Size Distribution

To understand the scale of objects in the dataset, bounding box areas were analyzed for all annotated objects in the training split. Bounding box area is computed as the product of bounding box width and height.

The dataset contains 1,286,871 bounding boxes. Summary statistics of the bounding box areas are shown in Table 2, and the distribution is visualized in Figure 2.
Table 2: Bounding Box Area Statistics (Train Split)
| Statistic          | Value           |
| ------------------ | --------------- |
| Count              | 1,286,871       |
| Mean               | 6,776 pixels²   |
| Standard Deviation | 22,850 pixels²  |
| Minimum            | 0.87 pixels²    |
| 25th Percentile    | 304 pixels²     |
| Median             | 817 pixels²     |
| 75th Percentile    | 3,022 pixels²   |
| Maximum            | 917,709 pixels² |

Figure 2 shows the distribution of bounding box areas using a logarithmic scale.

Observations

The distribution of bounding box areas is highly skewed toward small objects. The median bounding box area is only 817 pixels², meaning that half of the objects occupy less than this area in the image.

The mean area (6,776 pixels²) is significantly larger than the median, indicating the presence of a small number of very large objects that increase the average.

Further analysis shows that:

25% of objects occupy less than 304 pixels²

75% of objects are smaller than 3,022 pixels²

This indicates that most objects occupy relatively small regions of the image.

The smallest bounding boxes have an area of 0.87 pixels², which likely corresponds to extremely distant or partially visible objects. In contrast, the largest bounding boxes reach 917,709 pixels², typically corresponding to nearby vehicles such as buses or trucks.

Implications for Object Detection

The large number of small objects presents a significant challenge for object detection models. Small objects contain fewer pixels and are therefore harder to detect, particularly when feature maps are downsampled in deep convolutional networks.

This challenge is particularly relevant for classes such as traffic lights and traffic signs, which often appear far from the camera and therefore occupy very small regions of the image.

Modern detection architectures address this issue using techniques such as:

Feature Pyramid Networks (FPN) for multi-scale detection

higher-resolution feature maps

specialized small-object detection strategies

Understanding the size distribution of objects helps guide the design of detection models and training strategies for autonomous driving perception tasks.

5. . Aspect Ratio Analysis

To analyze the geometric characteristics of annotated objects, the aspect ratio of each bounding box was computed. The aspect ratio is defined as the ratio between the bounding box width and height:

Aspect Ratio
=
width
height
Aspect Ratio=
height
width
	​


Aspect ratio analysis helps identify the typical shapes of objects in the dataset and detect extreme bounding boxes that may affect detection performance.

The summary statistics of aspect ratios for the training split are shown in Table 3, and the distribution is visualized in Figure 3.

Table 3: Bounding Box Aspect Ratio Statistics (Train Split)
| Statistic          | Value     |
| ------------------ | --------- |
| Count              | 1,286,871 |
| Mean               | 1.22      |
| Standard Deviation | 0.95      |
| Minimum            | 0.0015    |
| 25th Percentile    | 0.74      |
| Median             | 1.10      |
| 75th Percentile    | 1.49      |
| Maximum            | 496.83    |


Figure 3 shows the distribution of bounding box aspect ratios.

Observations

The median aspect ratio is 1.10, indicating that most objects are slightly wider than they are tall. This is consistent with the dominance of vehicle classes such as cars, trucks, and buses, which typically have horizontally elongated shapes.

Approximately 50% of objects have aspect ratios between 0.74 and 1.49, suggesting that the majority of bounding boxes are close to square or moderately rectangular.

However, the dataset also contains several extreme aspect ratios. The smallest observed value is 0.0015, indicating extremely tall and narrow bounding boxes, while the maximum aspect ratio reaches 496.83, corresponding to extremely wide bounding boxes.

These extreme values may arise due to:

vertically elongated objects such as traffic lights

partially visible objects near image boundaries

annotation inconsistencies

Implications for Object Detection

Bounding box aspect ratios play an important role in object detection models that rely on anchor boxes, such as Faster R-CNN or YOLO. If anchor box shapes do not adequately represent the true distribution of object shapes, the detector may struggle to localize objects accurately.

The relatively concentrated aspect ratio distribution between 0.74 and 1.49 suggests that most objects can be represented using a limited set of anchor shapes. However, the presence of extreme aspect ratios indicates that some objects may still be difficult to capture using standard anchor configurations.

Understanding the aspect ratio distribution is therefore useful for designing anchor boxes and improving bounding box regression performance in object detection models.

6. Scene Density Analysis

In addition to object-level statistics, it is important to analyze the number of objects present in each image. This metric reflects the complexity of driving scenes and provides insight into how crowded the dataset is.

Scene density was computed by counting the number of annotated objects in each image of the training split.

The training split contains 69,863 images. The summary statistics for the number of objects per image are shown in Table 4, and the distribution is visualized in Figure 4.

Table 4: Objects per Image Statistics (Train Split)
| Statistic          | Value  |
| ------------------ | ------ |
| Count              | 69,863 |
| Mean               | 18.42  |
| Standard Deviation | 9.62   |
| Minimum            | 3      |
| 25th Percentile    | 11     |
| Median             | 17     |
| 75th Percentile    | 24     |
| Maximum            | 91     |

Figure 4 shows the distribution of the number of objects per image.

Observations

On average, each image contains approximately 18.4 annotated objects, indicating that driving scenes in the dataset typically contain multiple objects such as vehicles, pedestrians, and traffic infrastructure.

The median number of objects per image is 17, which suggests that most scenes contain a moderate number of objects.

However, the dataset also includes highly complex scenes. The most crowded images contain up to 91 objects, which likely correspond to dense urban environments with heavy traffic and pedestrian activity.

Further analysis shows that:

25% of images contain more than 24 objects

75% of images contain at least 11 objects

This indicates that crowded scenes are common in the dataset.

Implications for Object Detection

High scene density introduces several challenges for object detection models:

Occlusion: objects partially block each other

Overlapping bounding boxes

Visually cluttered environments

These challenges are particularly common in urban driving scenarios. Object detectors must therefore be capable of accurately identifying multiple objects in close proximity.

Understanding scene density helps evaluate the difficulty of the dataset and informs the design of detection models capable of handling crowded environments.

7. Small Object Analysis

Small objects are particularly challenging for object detection models because they occupy very few pixels in the image and contain limited visual information. To quantify this challenge, objects were classified as small objects if either their width or height was less than 10 pixels.

Using this criterion, 135,254 bounding boxes were identified as small objects in the training split.

Table 5: Small Object Statistics (Train Split)
Metric	Value
Total Objects	1,286,871
Small Objects (<10 px)	135,254
Small Object Ratio	10.51%
Observations

The analysis shows that approximately 10.51% of all annotated objects are small objects. This means that roughly one out of every ten objects occupies a very small region of the image.

Small objects typically correspond to objects that are far from the camera or naturally small in physical size. In autonomous driving scenarios, this often includes:

traffic lights

traffic signs

distant pedestrians

far-away vehicles

Because these objects appear far from the vehicle-mounted camera, they occupy only a few pixels in the image.

Implications for Object Detection

Small objects present several challenges for object detection models:

they contain limited visual detail

they may disappear in downsampled feature maps

they are more sensitive to image noise and compression

Detecting small objects reliably is crucial for autonomous driving applications, as traffic lights and road signs often fall into this category.

Modern object detection architectures address this challenge using techniques such as:

Feature Pyramid Networks (FPN) for multi-scale feature representation

higher-resolution feature maps

specialized small-object detection strategies

Understanding the proportion of small objects in the dataset helps guide the design of detection models that are robust to small-scale objects.

8. Spatial Distribution of Objects

To analyze where objects typically appear within images, the center coordinates of all bounding boxes were computed and visualized as a spatial heatmap. The resulting visualization is shown in Figure 5.

Figure 5: Spatial heatmap of bounding box centers in the training split.

Observations

The heatmap reveals a clear spatial concentration of objects near the center of the image, particularly around the horizontal midpoint of the frame. The highest density of object centers occurs approximately near x ≈ 600 pixels, which corresponds to the center of the camera's field of view.

Vertically, the highest concentration occurs around y ≈ 320–350 pixels, which corresponds to the region where the road and vehicles in front of the ego vehicle are typically visible.

The density gradually decreases toward the top and bottom edges of the image, indicating that objects are less frequently annotated near these regions.

This spatial pattern reflects the natural geometry of driving scenes:

Image Region	Typical Objects
Upper region	traffic lights, traffic signs
Middle region	vehicles and pedestrians
Lower region	nearby vehicles and road surface

The concentration around the center also indicates that the dataset primarily captures objects that are directly in front of the ego vehicle, which is consistent with forward-facing camera placement in autonomous driving systems.

Implications for Object Detection

The spatial distribution reveals a form of location bias in the dataset. Because objects appear more frequently near the center of the image, detection models may implicitly learn that important objects are likely to appear in this region.

While this bias can help models learn useful contextual cues, it may also reduce generalization if objects appear in unusual locations or if the camera viewpoint changes.

Understanding these spatial patterns is therefore useful for evaluating dataset bias and designing detection models that remain robust under varying camera perspectives.


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
