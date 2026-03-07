"""
Swin Transformer Tiny + Faster R-CNN model.

This module builds an object detection model using
Swin Transformer Tiny as backbone and Faster R-CNN
as the detection head.
"""

import torch
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from timm import create_model

from src.config import DETECTION_CLASSES


def build_swin_backbone():
    """
    Create Swin-T backbone and adapt it for FasterRCNN.

    Returns
    -------
    backbone : torch.nn.Module
    """

    # Load Swin Transformer Tiny
    swin = create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        features_only=True
    )

    # Extract feature channels
    in_channels_list = swin.feature_info.channels()

    backbone = BackboneWithFPN(
        swin,
        return_layers={
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
        },
        in_channels_list=in_channels_list,
        out_channels=256,
    )

    return backbone


def build_model():
    """
    Build Swin-T FasterRCNN model for BDD100K.

    Returns
    -------
    model : FasterRCNN
    """

    backbone = build_swin_backbone()

    num_classes = len(DETECTION_CLASSES) + 1

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
    )

    return model