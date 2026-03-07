import torch
import torch.nn as nn

from collections import OrderedDict

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

from timm import create_model

from src.config import DETECTION_CLASSES


class SwinBackboneWithFPN(nn.Module):

    def __init__(self):
        super().__init__()

        # Swin backbone
        self.body = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        # remove fixed input size constraint
        self.body.patch_embed.img_size = None

        # channel sizes from Swin stages
        in_channels_list = self.body.feature_info.channels()

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256
        )

        self.out_channels = 256

    def forward(self, x):

        features = self.body(x)

        features = {
            str(i): f for i, f in enumerate(features)
        }

        features = self.fpn(features)

        return features


def build_model():

    backbone = SwinBackboneWithFPN()

    num_classes = len(DETECTION_CLASSES) + 1

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
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

    return model