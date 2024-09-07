import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from cv.utils import MetaWrapper


class VGG16(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for VGG16 architecture from paper on: Very Deep Convolutional Networks for Large-Scale Image Recognition"

    def __init__(self, num_classes, in_channels=3):
        super(VGG16, self).__init__()

        self.feature_extractor_layers = OrderedDict(
            [
                ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding="same", stride=1)),
                ("relu1", nn.ReLU(inplace=True)),
                
                ("conv2", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same", stride=1)),
                ("relu2", nn.ReLU(inplace=True)),

                ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),

                ("conv3", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same", stride=1)),
                ("relu3", nn.ReLU(inplace=True)),

                ("conv4", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same", stride=1)),
                ("relu4", nn.ReLU(inplace=True)),

                ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),

                ("conv5", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same", stride=1)),
                ("relu5", nn.ReLU(inplace=True)),

                ("conv6", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same", stride=1)),
                ("relu6", nn.ReLU(inplace=True)),

                ("conv7", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same", stride=1)),
                ("relu7", nn.ReLU(inplace=True)),

                ("pool3", nn.MaxPool2d(kernel_size=2, stride=2)),

                ("conv8", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu8", nn.ReLU(inplace=True)),

                ("conv9", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu9", nn.ReLU(inplace=True)),

                ("conv10", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu10", nn.ReLU(inplace=True)),

                ("pool4", nn.MaxPool2d(kernel_size=2, stride=2)),

                ("conv11", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu11", nn.ReLU(inplace=True)),

                ("conv12", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu12", nn.ReLU(inplace=True)),

                ("conv13", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same", stride=1)),
                ("relu13", nn.ReLU(inplace=True)),

                ("pool5", nn.MaxPool2d(kernel_size=2, stride=2))
            ]
        )

        self.feature_extractor = nn.Sequential(self.feature_extractor_layers)

        self.classifier_layers = OrderedDict(
            [
                ("dropout1", nn.Dropout(p=0.5)),
                ("fc1", nn.Linear(in_features=512*7*7, out_features=4096)),
                ("relu_fc1", nn.ReLU(inplace=True)),
                ("dropout2", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(in_features=4096, out_features=4096)),
                ("relu_fc2", nn.ReLU(inplace=True)),
                ("fc_class", nn.Linear(in_features=4096, out_features=num_classes))
            ]
        )

        self.classifier = nn.Sequential(self.classifier_layers)

        self.initialize()

        self.feature_extractor_1 = self.feature_extractor[:10]
        self.feature_extractor_2 = self.feature_extractor[10:]

    def vgg_init(self, convLayer):

        init_n = (convLayer.kernel_size[0] ** 2) * convLayer.out_channels
        init_mean = 0.0
        init_std = np.sqrt(2 / init_n)

        nn.init.normal_(convLayer.weight, mean=init_mean, std=init_std)
        nn.init.constant_(convLayer.bias, val=0.0)

    def initialize(self):
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.Conv2d):
                self.vgg_init(module)

    def forward(self, x):

        x = self.feature_extractor_1(x)
        x = self.feature_extractor_2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        
        return x
    