from collections import OrderedDict
from turtle import forward
import torch
import torch.nn as nn

from src.gpu_devices import GPU_Support


class VGG16(nn.Module):

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

        self.feature_extractor_1 = self.feature_extractor[:10]
        self.feature_extractor_2 = self.feature_extractor[10:]

        if GPU_Support.support_gpu == 2:
            self.feature_extractor_1.to("cuda:0")
            self.feature_extractor_2.to("cuda:1")
            self.classifier.to("cuda:1")
        elif GPU_Support.support_gpu == 1:
            self.feature_extractor.to("cuda:0")
            self.classifier.to("cuda:0")

    def forward(self, x):

        feature_extractor_1_device = next(self.feature_extractor_1.parameters()).device
        x = x.cpu()
        x = x.to(feature_extractor_1_device)
        x = self.feature_extractor_1(x)

        feature_extractor_2_device = next(self.feature_extractor_2.parameters()).device
        x = x.cpu()
        x = x.to(feature_extractor_2_device)
        x = self.feature_extractor_2(x)

        x = torch.flatten(x, start_dim=1)

        classifier_device = next(self.classifier.parameters()).device
        x = x.cpu()
        x = x.to(classifier_device)
        x = self.classifier(x)

        return x
    