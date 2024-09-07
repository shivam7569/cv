import torch
import torch.nn as nn

from cv.backbones import VGG16
from cv.src.checkpoints import Checkpoint
from cv.utils import MetaWrapper


class DeepLabv1(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for DeepLabv1 architecture from paper on: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"

    def __init__(self, num_classes, backbone_params, load_weights, dropout_rate=0.5):
        
        super(DeepLabv1, self).__init__()

        backbone = VGG16(**backbone_params)
        if load_weights:
            checkpoint = Checkpoint.load(
                model=backbone,
                name="VGG16",
                return_checkpoint=True
            )
            feature_extractor_weights = {k.replace("feature_extractor.", ''): v for k, v in checkpoint["model_state_dict"].items() if "feature_extractor" in k and "_1" not in k and "_2" not in k}
            backbone.feature_extractor.load_state_dict(feature_extractor_weights)

        for name, module in backbone.feature_extractor.named_modules():
            if "pool" in name:
                module.kernel_size = (3, 3)
                module.padding = 1

                if ("4" in name) or ("5" in name):
                    module.stride = 1

            if "conv" in name and (("11" in name) or ("12" in name) or ("13" in name)):
                module.dilation = (2, 2)

        backbone.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=12,
                dilation=12
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(
                in_channels=1024,
                out_channels=num_classes,
                kernel_size=1,
                stride=1
            )
        )

        self.feature_extractor = backbone.feature_extractor
        self.classifier = backbone.classifier

        ch = torch.load("/media/drive6/hqh2kor/projects/cv/archives/checkpoints/DeepLabv1/epoch_checkpoint.pth")["model_state_dict"]
        f_ch = {k.replace("feature_extractor.", ''):v for k, v in ch.items() if "feature_extractor" in k}
        c_ch = {k.replace("classifier.", ''):v for k, v in ch.items() if "classifier" in k}

        self.feature_extractor.load_state_dict(f_ch)
        self.classifier.load_state_dict(c_ch)

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)

        return logits
