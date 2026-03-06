"""
Faster R-CNN model definition for BDD100K object detection.

This module creates a Faster R-CNN model with a ResNet50 backbone
and Feature Pyramid Network (FPN). The classifier head is replaced
to match the BDD100K detection classes.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes: int):
    """
    Create a Faster R-CNN model with a ResNet50-FPN backbone.

    Parameters
    ----------
    num_classes : int
        Number of classes including background.

    Returns
    -------
    model : torchvision.models.detection.FasterRCNN
        Configured Faster R-CNN model ready for training.
    """

    # Load pretrained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )

    # Get number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace classifier head
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model


def build_model(num_detection_classes: int):
    """
    Build Faster R-CNN model for BDD100K.

    Parameters
    ----------
    num_detection_classes : int
        Number of object detection classes (excluding background).

    Returns
    -------
    model : FasterRCNN
        Detection model ready for training.
    """

    # Add background class
    num_classes = num_detection_classes + 1

    model = get_faster_rcnn_model(num_classes)

    return model