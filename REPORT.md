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

**Table 1: Object Instance Distribution (Train Split)**
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

Figure 1 shows the class distribution using a logarithmic scale to better visualize rare classes.
### Figure 1: Class Distribution (Train Split)

![Class Distribution](outputs/figures/class_distribution_train.png)

Observations (from Class Distribution)

### Observations

- The dataset shows **significant class imbalance**, with a few classes dominating the distribution.
- The **car class accounts for 55.42%** of all annotated objects, making it the most frequent category.
- **Traffic signs (18.62%) and traffic lights (14.46%)** are also highly represented in the dataset.
- The three most frequent classes (**car, traffic sign, traffic light**) together represent **88.5% of all objects**, indicating a strong **long-tail distribution**.
- Several classes are **extremely underrepresented**, including **motor (0.23%)**, **rider (0.35%)**, and **train (0.01%)**.
- The **train class appears only 136 times**, meaning cars appear **over 5,200 times more frequently than trains**.

### Implications for Object Detection

- Object detection models trained on this dataset may become **biased toward dominant classes**, particularly cars.
- Rare classes such as **train, rider, and motor** may have **lower detection accuracy** due to limited training examples.
- The training loss may be **dominated by frequent classes**, making it harder for the model to learn representations for rare categories.
- The model may learn **strong priors for common objects**, potentially reducing sensitivity to rare but important objects.
- Techniques such as **class-balanced sampling, focal loss, or targeted data augmentation** may be required to improve detection performance for rare classes.


### 3.2 Train vs Validation Split Analysis

To ensure that the validation dataset is representative of the training data, the class distribution of object instances was compared across the two splits.

### Class Distribution Comparison

The validation dataset contains **185,526 annotated objects** across the same 10 detection classes.  
Table 6 compares the class distributions between the training and validation splits.

**Table 6: Class Distribution Comparison (Train vs Validation)**

| Class | Train Count | Train % | Val Count | Val % |
|------|-------------|--------|-----------|-------|
| car | 713,211 | 55.42% | 102,506 | 55.24% |
| traffic sign | 239,686 | 18.62% | 34,908 | 18.82% |
| traffic light | 186,117 | 14.46% | 26,885 | 14.49% |
| person | 91,349 | 7.10% | 13,262 | 7.15% |
| truck | 29,971 | 2.33% | 4,245 | 2.29% |
| bus | 11,672 | 0.91% | 1,597 | 0.86% |
| bike | 7,210 | 0.56% | 1,007 | 0.54% |
| rider | 4,517 | 0.35% | 649 | 0.35% |
| motor | 3,002 | 0.23% | 452 | 0.24% |
| train | 136 | 0.01% | 15 | 0.01% |

### Observations

- The validation split follows **similar class distribution trends** as the training dataset.
- The **car class remains the dominant category** in both splits.
- **Traffic signs and traffic lights** are also among the most frequent classes in both datasets.
- Rare classes such as **train, motor, and rider** remain highly underrepresented.
- The **train class appears only 15 times in the validation set**, which further emphasizes its rarity.

### Implications for Model Evaluation

- Since the validation distribution closely matches the training distribution, the validation set is **representative of the training data**.
- Performance metrics on the validation dataset are therefore **likely to reflect real training behavior**.
- However, evaluation for **rare classes may be unstable**, as very few validation examples exist for these categories.
- Metrics for classes such as **train, motor, and rider** should therefore be interpreted with caution.


---
### 3.3. Bounding Box Size Distribution

To understand the scale of objects in the dataset, **bounding box areas** were analyzed for all annotated objects in the training split. The bounding box area is computed as the product of the **bounding box width and height**.

The dataset contains **1,286,871 bounding boxes**. Summary statistics of the bounding box areas are shown in **Table 2**, and the distribution is visualized in **Figure 2**.

**Table 2: Bounding Box Area Statistics (Train Split)**
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

### Observations

- The distribution of bounding box areas is **highly skewed toward small objects**.
- The **median area is 817 pixels²**, meaning that half of the objects occupy less than this area.
- The **mean area (6,776 pixels²)** is much larger than the median, indicating the presence of a few **very large objects** that increase the average.
- **25% of objects are smaller than 304 pixels²**, and **75% are smaller than 3,022 pixels²**, showing that most objects occupy relatively small image regions.
- Bounding box areas range from **0.87 pixels² to 917,709 pixels²**, reflecting a large variation in object scales.

### Implications for Object Detection

- The high proportion of **small objects makes detection more challenging**, as they contain fewer pixels and visual details.
- Small objects are more likely to be lost in **downsampled feature maps** in deep neural networks.
- This issue is particularly important for **traffic lights and traffic signs**, which often appear far from the camera.
- Techniques such as **Feature Pyramid Networks (FPN)**, **higher-resolution feature maps**, and **multi-scale detection strategies** are commonly used to improve detection performance for small objects.

### 3.4. Aspect Ratio Analysis

To analyze the geometric characteristics of annotated objects, the **aspect ratio** of each bounding box was computed. The aspect ratio is defined as the ratio between the bounding box width and height:

$$
\text{Aspect Ratio} = \frac{\text{width}}{\text{height}}
$$

Aspect ratio analysis helps identify the **typical shapes of objects in the dataset** and detect **extreme bounding boxes** that may affect detection performance.

The summary statistics of aspect ratios for the training split are shown in **Table 3**, and the distribution is visualized in **Figure 3**.

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

### Observations

- The **median aspect ratio is 1.10**, indicating that most objects are slightly wider than they are tall.
- Approximately **50% of objects have aspect ratios between 0.74 and 1.49**, meaning most bounding boxes are close to square or moderately rectangular.
- The **mean aspect ratio (1.22)** is close to the median, suggesting that the majority of object shapes are relatively consistent.
- The dataset also contains **extreme aspect ratios**, ranging from **0.0015 to 496.83**, indicating the presence of very tall or very wide bounding boxes.
- These extreme values may arise from **elongated objects (e.g., traffic lights)**, **partially visible objects near image boundaries**, or **annotation inconsistencies**.

### Implications for Object Detection

- Aspect ratio distribution is important for **anchor-based detectors** such as Faster R-CNN or YOLO.
- If anchor box shapes do not match the **true object shape distribution**, the model may struggle to localize objects accurately.
- Since most aspect ratios fall between **0.74 and 1.49**, a **limited set of anchor shapes** may capture the majority of objects effectively.
- However, **extreme aspect ratios** may still be difficult for standard anchors to represent, potentially affecting detection performance.
- Understanding aspect ratio distribution helps guide **anchor design and bounding box regression strategies** in object detection models.

### 3.5 Scene Density Analysis

In addition to object-level statistics, it is important to analyze the **number of objects present in each image**. This metric reflects the **complexity of driving scenes** and provides insight into how crowded the dataset is.

Scene density was computed by counting the number of annotated objects in each image of the **training split**.

The training split contains **69,863 images**. The summary statistics for the number of objects per image are shown in **Table 4**, and the distribution is visualized in **Figure 4**.

**Table 4: Objects per Image Statistics (Train Split)**
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

### Observations

- Each image contains on average **18.4 annotated objects**, indicating that most driving scenes contain multiple objects such as vehicles, pedestrians, and traffic infrastructure.
- The **median number of objects per image is 17**, suggesting that most scenes have a moderate level of complexity.
- Some scenes are highly dense, with **up to 91 objects in a single image**, likely corresponding to crowded urban environments.
- **25% of images contain more than 24 objects**, while **75% contain at least 11 objects**, indicating that crowded scenes are relatively common.

### Implications for Object Detection

- High scene density introduces challenges such as:
  - **Occlusion**, where objects partially block each other.
  - **Overlapping bounding boxes** between nearby objects.
  - **Visually cluttered scenes**, which make object boundaries harder to distinguish.
- Object detection models must therefore be capable of **detecting multiple objects in close proximity**.
- Understanding scene density helps evaluate the **difficulty of the dataset** and guides the design of models that can handle **complex urban driving environments**.

### 3.6 Small Object Analysis

Small objects are particularly challenging for object detection models because they occupy very few pixels in the image and contain limited visual information. To quantify this challenge, objects were classified as **small objects** if either their width or height was less than **10 pixels**.

Using this criterion, **135,254 bounding boxes** were identified as small objects in the training split.

**Table 5: Small Object Statistics (Train Split)**

| Metric | Value |
|------|------|
| Total Objects | 1,286,871 |
| Small Objects (<10 px) | 135,254 |
| Small Object Ratio | 10.51% |

### Observations

- Approximately **10.51% of all annotated objects are small objects**, meaning roughly **one out of every ten objects** occupies a very small region of the image.
- Small objects often correspond to objects that are **far from the camera** or naturally small in size.
- In autonomous driving scenes, small objects frequently include:
  - traffic lights  
  - traffic signs  
  - distant pedestrians  
  - far-away vehicles
- These objects occupy only a **few pixels in the image**, making them difficult to detect.

### Implications for Object Detection

- Small objects contain **limited visual detail**, which makes detection more challenging.
- They may disappear in **downsampled feature maps** in deep neural networks.
- Small objects are more sensitive to **image noise and compression artifacts**.
- Reliable detection of small objects is critical for autonomous driving systems, since **traffic lights and road signs** often fall into this category.
- Modern detection architectures address this challenge using techniques such as:
  - **Feature Pyramid Networks (FPN)** for multi-scale feature representation
  - **higher-resolution feature maps**
  - **specialized small-object detection strategies**

Understanding the proportion of small objects in the dataset helps guide the design of detection models that are robust to **small-scale objects**.


### 3.7 Spatial Distribution of Objects

To analyze where objects typically appear within images, the **center coordinates of all bounding boxes** were computed and visualized as a **spatial heatmap**. The resulting visualization is shown in **Figure 5**.

**Figure 5: Spatial heatmap of bounding box centers in the training split.**

### Observations

- The heatmap shows a **strong spatial concentration of objects near the center of the image**, particularly around the horizontal midpoint.
- The highest density occurs around **x ≈ 600 pixels**, corresponding to the center of the camera's field of view.
- Vertically, the highest concentration occurs around **y ≈ 320–350 pixels**, which represents the region where the road and vehicles in front of the ego vehicle are typically visible.
- Object density gradually **decreases toward the top and bottom edges of the image**, indicating fewer annotated objects in these areas.
- This spatial pattern reflects the natural geometry of driving scenes:

| Image Region | Typical Objects |
|--------------|----------------|
| Upper region | traffic lights, traffic signs |
| Middle region | vehicles, pedestrians |
| Lower region | nearby vehicles, road surface |

- The central concentration also indicates that most annotated objects are **directly in front of the ego vehicle**, consistent with the placement of forward-facing cameras in autonomous driving systems.

### Implications for Object Detection

- The spatial distribution introduces **location bias** in the dataset.
- Detection models may implicitly learn that important objects are **more likely to appear near the center of the image**.
- While this bias can help models learn contextual cues, it may **reduce generalization** if objects appear in unusual locations.
- Understanding spatial patterns helps identify dataset biases and supports the design of models that remain **robust to different camera viewpoints and object positions**.


## 3.8 Annotation Anomalies

During dataset analysis, several potential annotation anomalies and edge cases were identified. These anomalies can impact model training and evaluation if not properly handled.

### Extreme Aspect Ratios

Bounding boxes with extremely large or small aspect ratios were observed in the dataset.

- Aspect ratios greater than **5** or smaller than **0.2** were flagged as potential anomalies.
- These cases may correspond to:
  - vertically elongated objects such as **traffic lights**
  - partially visible objects near image boundaries
  - possible annotation inconsistencies.

Such extreme shapes may make it difficult for anchor-based detectors to match suitable anchor boxes.

### Extremely Small Bounding Boxes

Some bounding boxes have very small dimensions, with areas approaching **1 pixel²**.

These cases typically correspond to:

- distant objects
- partially occluded objects
- small infrastructure elements such as traffic lights or signs.

Small objects contain limited visual information and are therefore harder for detection models to recognize.

### Crowded Scenes

Certain images contain **very high numbers of annotated objects**, with some scenes containing **up to 91 objects**.

These crowded scenes introduce challenges such as:

- object occlusion
- overlapping bounding boxes
- visually cluttered environments.

Such cases can increase the difficulty of object detection and may lead to missed detections.

### Impact on Model Training

These anomalies highlight potential challenges for object detection models:

- extreme bounding box shapes may affect anchor matching
- small objects are harder to detect in downsampled feature maps
- crowded scenes increase occlusion and overlapping detections.

Understanding these cases helps ensure that the dataset is properly interpreted before training detection models.

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
