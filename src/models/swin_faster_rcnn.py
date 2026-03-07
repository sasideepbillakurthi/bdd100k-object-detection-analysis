import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from timm import create_model

from src.config import DETECTION_CLASSES


# -----------------------------
# Class Weights (BDD100K)
# -----------------------------
CLASS_WEIGHTS = torch.tensor([
    1.0,   # background
    0.2,   # traffic light
    0.2,   # traffic sign
    0.05,  # car
    0.3,   # person
    0.4,   # bus
    0.3,   # truck
    0.6,   # rider
    0.6,   # bike
    0.7,   # motor
    1.0    # train
])


# -----------------------------
# Custom ROI Loss
# -----------------------------
def weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    weights = CLASS_WEIGHTS.to(class_logits.device)

    classification_loss = F.cross_entropy(
        class_logits,
        labels,
        weight=weights
    )

    sampled_pos_inds_subset = torch.where(labels > 0)[0]

    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, num_classes, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )

    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


# -----------------------------
# Swin Backbone
# -----------------------------
class SwinBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        # Disable fixed input size
        self.backbone.patch_embed.img_size = None

        in_channels_list = self.backbone.feature_info.channels()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool()
        )

        self.out_channels = 256

    def forward(self, x):

        features = self.backbone(x)

        # Convert NHWC -> NCHW
        features = {
            str(i): f.permute(0, 3, 1, 2)
            for i, f in enumerate(features)
        }

        features = self.fpn(features)

        return features


# -----------------------------
# Build Model
# -----------------------------
def build_model():

    backbone = SwinBackbone()

    num_classes = len(DETECTION_CLASSES) + 1

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3.0),) * 5,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # Override classification loss
    model.roi_heads.fastrcnn_loss = weighted_fastrcnn_loss

    return model