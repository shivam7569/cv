import math
import torch
import torch.nn as nn
from collections import OrderedDict

from cv.utils import MetaWrapper

class SPPNet(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for SPPNet architecture from paper on: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    
    def __init__(self, num_classes, in_channels=3, levels=[3, 2, 1]):
        
        super(SPPNet, self).__init__()

        self.levels = levels

        self.feature_extractor_layers = OrderedDict(
            [
                ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4)),
                ("relu1", nn.ReLU(inplace=True)),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2)),

                ("conv2", nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(kernel_size=3, stride=2)),

                ("conv3", nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)),
                ("relu3", nn.ReLU(inplace=True)),

                ("conv4", nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)),
                ("relu4", nn.ReLU(inplace=True)),

                ("conv5", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)),
                ("relu5", nn.ReLU(inplace=True))
            ]
        )

        self.feature_extractor = nn.Sequential(self.feature_extractor_layers)

        self.classifier_layers = OrderedDict(
            [
                ("dropout1", nn.Dropout(p=0.5)),
                ("fc1", nn.Linear(256*2*7, 4096)),
                ("relu_fc1", nn.ReLU(inplace=True)),
                ("dropout2", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(4096, 4096)),
                ("relu_fc2", nn.ReLU(inplace=True)),
                ("fc_class", nn.Linear(4096, num_classes))
            ]
        )

        self.classifier = nn.Sequential(self.classifier_layers)

    
    def _spp_pyramid(self, x):
        a = x.shape[-1]
        spp_layer_outs = [torch.flatten(self._spp_level(a=a, n=i)(x), start_dim=1) for i in self.levels]
        spp_out = torch.cat(spp_layer_outs, dim=1)

        return spp_out

    def _spp_level(self, a, n):
        window_size = math.ceil(a/n)
        stride = math.floor(a/n)

        max_pool = nn.MaxPool2d(kernel_size=window_size, stride=stride)

        return max_pool

    def forward(self, x):
        x = self.feature_extractor(x)
        spp_out = self._spp_pyramid(x)
        logits = self.classifier(spp_out)

        return logits
