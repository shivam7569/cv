import numpy as np
import torch.nn as nn

from cv.utils import MetaWrapper

class MobileNet(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for MobileNet-v1 architecture from paper on: Efficient Convolutional Neural Networks for Mobile Vision Applications"

    def __init__(self, num_classes, in_channels=3, alpha=1):
        super(MobileNet, self).__init__()

        channel_lambda = lambda x: int(alpha * x)

        self.init_features = ConvBlock(
            in_channels=in_channels, out_channels=channel_lambda(32),
            kernel_size=3, stride=2, padding=1, groups=1
        )
        self.mobileNet_block_1 = MobileNetBlock(
            in_channels=channel_lambda(32), out_channels=channel_lambda(64),
            kernel_size=3, stride=1, padding=1
        )
        self.mobileNet_block_2 = MobileNetBlock(
            in_channels=channel_lambda(64), out_channels=channel_lambda(128),
            kernel_size=3, stride=2, padding=1
        )
        self.mobileNet_block_3 = MobileNetBlock(
            in_channels=channel_lambda(128), out_channels=channel_lambda(128),
            kernel_size=3, stride=1, padding=1
        )
        self.mobileNet_block_4 = MobileNetBlock(
            in_channels=channel_lambda(128), out_channels=channel_lambda(256),
            kernel_size=3, stride=2, padding=1
        )
        self.mobileNet_block_5 = MobileNetBlock(
            in_channels=channel_lambda(256), out_channels=channel_lambda(256),
            kernel_size=3, stride=1, padding=1
        )
        self.mobileNet_block_6 = MobileNetBlock(
            in_channels=channel_lambda(256), out_channels=channel_lambda(512),
            kernel_size=3, stride=2, padding=1
        )
        self.mobileNet_blockList = nn.Sequential(
            *[MobileNetBlock(
                in_channels=channel_lambda(512), out_channels=channel_lambda(512),
                kernel_size=3, stride=1, padding=1
            ) for _ in range(5)]
        )
        self.mobileNet_block_7 = MobileNetBlock(
            in_channels=channel_lambda(512), out_channels=channel_lambda(1024),
            kernel_size=3, stride=2, padding=1
        )
        self.mobileNet_block_8 = MobileNetBlock(
            in_channels=channel_lambda(1024), out_channels=channel_lambda(1024),
            kernel_size=3, stride=1, padding=1
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(channel_lambda(1024), num_classes)
        )

    def forward(self, x):
        x = self.init_features(x)
        x = self.mobileNet_block_1(x)
        x = self.mobileNet_block_2(x)
        x = self.mobileNet_block_3(x)
        x = self.mobileNet_block_4(x)
        x = self.mobileNet_block_5(x)
        x = self.mobileNet_block_6(x)
        x = self.mobileNet_blockList(x)
        x = self.mobileNet_block_7(x)
        x = self.mobileNet_block_8(x)
        x = self.classifier(x)

        return x

class MobileNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                kernel_size, stride, padding):
        
        super(MobileNetBlock, self).__init__()

        self.depth_wise = ConvBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        self.point_wise = ConvBlock(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.initializeConv(kernel_size=kernel_size, channels=out_channels)

    def initializeConv(self, kernel_size, channels):
        init_n = (kernel_size ** 2) * channels
        init_mean = 0.0
        init_std = np.sqrt(2 / init_n)
        
        nn.init.normal_(self.conv.weight, mean=init_mean, std=init_std)
        nn.init.constant_(self.conv.bias, val=0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
