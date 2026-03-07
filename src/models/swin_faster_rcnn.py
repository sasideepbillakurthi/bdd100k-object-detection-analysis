"""
Swin Transformer Tiny + Faster R-CNN model.

This module builds an object detection model using
Swin Transformer Tiny as backbone and Faster R-CNN
as the detection head.
"""

import torch
import torch.nn as nn

from collections import OrderedDict

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from timm import create_model

from src.config import DETECTION_CLASSES


class SwinBackbone(nn.Module):
    """
    Wrapper for Swin Transformer Tiny to make it compatible
    with torchvision detection models.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained Swin-T backbone
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            dynamic_img_size=True,
            img_size=(448, 448)
        )

        # Swin feature channels
        # Stage outputs: [96, 192, 384, 768]
        self.out_channels = 768

    def forward(self, x):
        """
        Forward pass through Swin backbone.

        Returns
        -------
        OrderedDict of feature maps
        """

        features = self.backbone(x)

        out = OrderedDict()

        for i, f in enumerate(features):
            out[str(i)] = f

        return out


def build_swin_backbone():
    """
    Create Swin backbone for FasterRCNN.

    Returns
    -------
    backbone : nn.Module
    """

    backbone = SwinBackbone()

    return backbone


def build_model():
    """
    Build Swin Transformer Tiny + FasterRCNN model.

    Returns
    -------
    model : FasterRCNN
    """

    backbone = build_swin_backbone()

    num_classes = len(DETECTION_CLASSES) + 1

    # Anchor generator for 4 feature maps
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = FasterRCNN(
    backbone=backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    
    # 1. Image Resizing: Forces input to 224x224 to match Swin's expectations
    min_size=448,
    max_size=448,
    
    # 2. Normalization: Uses ImageNet stats (Swin was trained with these)
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

    return model