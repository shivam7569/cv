import torch.nn as nn
from collections import OrderedDict

from cv import backbones
from cv.src.checkpoints import Checkpoint
from cv.utils import MetaWrapper

class FCN(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for FCN architecture from paper on: Fully Convolutional Networks for Semantic Segmentation"

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

        self.classifier_first = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )
        self.classifier_middle = nn.Conv2d(
            in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0
        )
        self.classifier_last = nn.Sequential(classifier_layers)

        self.convTranspose_first = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4
        )
        self.convTranspose_middle = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1
        )
        self.convTranspose_last = nn.ConvTranspose2d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1
        )

        self.initialize()

    def kernel_init(self, convLayer):
        nn.init.kaiming_normal_(convLayer.weight)
        nn.init.zeros_(convLayer.bias)

    def initialize(self):
        for module in self.classifier_last.modules():
            if isinstance(module, nn.Conv2d):
                self.kernel_init(module)

        nn.init.kaiming_normal_(self.classifier_first.weight)
        nn.init.zeros_(self.classifier_first.bias)
        nn.init.kaiming_normal_(self.classifier_middle.weight)
        nn.init.zeros_(self.classifier_middle.bias)

    
    def forward(self, x):
        pool_outs = []
        for pool_block in self.pool_blocks:
            x = pool_block.to(x.device)(x)
            pool_outs.append(x)

        stream_first = self.classifier_first(pool_outs[0])
        stream_middle = self.classifier_middle(pool_outs[1])
        stream_last = self.classifier_last(pool_outs[2])

        last_and_middle = stream_middle + self.convTranspose_last(stream_last)
        middle_and_first = stream_first + self.convTranspose_middle(last_and_middle)
        logits = self.convTranspose_first(middle_and_first)

        return logits
