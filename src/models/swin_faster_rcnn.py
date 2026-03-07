import torch
import torch.nn as nn

from collections import OrderedDict

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from timm import create_model

from src.config import DETECTION_CLASSES


class SwinBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        # Disable fixed input size restriction
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


def build_model():

    backbone = SwinBackbone()

    num_classes = len(DETECTION_CLASSES) + 1

    anchor_generator = AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),
    aspect_ratios=((0.33,0.5,1.0,2.0,3.0),)*5,
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