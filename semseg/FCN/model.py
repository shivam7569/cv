from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
from configs.config import setup_config
from src.checkpoints import Checkpoint
from utils.global_params import Global

class FCN(nn.Module):

    def __init__(
            self,
            backbone_name,
            backbone_params="default",
            num_classes=81
        ):

        super(FCN, self).__init__()

        if isinstance(backbone_params, str) and backbone_params == "default": backbone_params = {"num_classes": 1000}
        backbone_model = getattr(backbones, backbone_name)(**backbone_params)
        backbone_model = Checkpoint.load(
            model=backbone_model,
            name=backbone_name
        )
        if backbone_name == "VGG16":
            backbone = backbone_model.feature_extractor
        else:
            backbone = backbone_model.backbone

        self.pool_blocks = [
            backbone[:17],
            backbone[17:24],
            backbone[24:]
        ]

        classifier_layers = OrderedDict(
            [
                ("classifier_conv1", nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, stride=1, padding=3)),
                ("relu_classifier_conv1", nn.ReLU(inplace=True)),
                ("dropout1", nn.Dropout(p=0.5)),
                ("classifier_conv2", nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0)),
                ("relu_fc2", nn.ReLU(inplace=True)),
                ("dropout2", nn.Dropout(p=0.5)),
                ("class_mask_conv", nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, stride=1, padding=0))
            ]
        )

        self.classifier = nn.Sequential(classifier_layers)

        self.pointwise_classifier_1 = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )
        self.pointwise_classifier_2 = nn.Conv2d(
            in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )

        self.convTranspose_1 = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4
        )
        self.convTranspose_2 = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1
        )
        self.convTranspose_3 = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1
        )

        self.initialize()

    def kernel_init(self, convLayer):
        nn.init.zeros_(convLayer.weight)
        nn.init.zeros_(convLayer.bias)

    def initialize(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Conv2d):
                self.kernel_init(module)
    
    def forward(self, x):
        pool_outs = []
        for pool_block in self.pool_blocks:
            x = pool_block.to(x.device)(x)
            pool_outs.append(x)

        stream_1 = self.pointwise_classifier_1(pool_outs[0])
        stream_2 = self.pointwise_classifier_2(pool_outs[1])
        stream_3 = self.classifier(pool_outs[2])

        add_stream_3_and_stream_2 = stream_2 + self.convTranspose_3(stream_3)
        add_stream_3_and_stream_2_and_stream_1 = stream_1 + self.convTranspose_2(add_stream_3_and_stream_2)
        mask_out = self.convTranspose_1(add_stream_3_and_stream_2_and_stream_1)

        return mask_out