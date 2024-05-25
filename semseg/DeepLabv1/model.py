import torch.nn as nn
from backbones import VGG16
from src.checkpoints import Checkpoint


class DeepLabv1(nn.Module):

    def __init__(self, num_classes, backbone_params, dropout_rate=0.5):
        
        super(DeepLabv1, self).__init__()

        backbone = VGG16(**backbone_params)
        backbone = Checkpoint.load(
            model=backbone,
            name="VGG16"
        )

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

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)

        return logits
