import numpy as np
import torch
import torch.nn as nn

from cv.utils import MetaWrapper

class MobileNetv2(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for MobileNet-v2 architecture from paper on: Inverted Residuals and Linear Bottlenecks"

    def __init__(self, num_classes, in_channels=3, expansion_rate=6, alpha=1):
        super(MobileNetv2, self).__init__()

        channel_lambda = lambda x: int(alpha * x)

        self.init_features = ConvBlock(
            in_channels=in_channels, out_channels=channel_lambda(32),
            kernel_size=3, stride=2, padding=1, groups=1, activation=True
        )
        self.bottleneck_1 = BottleNeckLayer(
            num_blocks=1, in_channels=channel_lambda(32),
            out_channels=channel_lambda(16), stride=1, expansion_rate=1
        )
        self.bottleneck_2 = BottleNeckLayer(
            num_blocks=2, in_channels=channel_lambda(16),
            out_channels=channel_lambda(24), stride=2, expansion_rate=expansion_rate
        )
        self.bottleneck_3 = BottleNeckLayer(
            num_blocks=3, in_channels=channel_lambda(24),
            out_channels=channel_lambda(32), stride=2, expansion_rate=expansion_rate
        )
        self.bottleneck_4 = BottleNeckLayer(
            num_blocks=4, in_channels=channel_lambda(32),
            out_channels=channel_lambda(64), stride=2, expansion_rate=expansion_rate
        )
        self.bottleneck_5 = BottleNeckLayer(
            num_blocks=3, in_channels=channel_lambda(64),
            out_channels=channel_lambda(96), stride=1, expansion_rate=expansion_rate
        )
        self.bottleneck_6 = BottleNeckLayer(
            num_blocks=3, in_channels=channel_lambda(96),
            out_channels=channel_lambda(160), stride=2, expansion_rate=expansion_rate
        )
        self.bottleneck_7 = BottleNeckLayer(
            num_blocks=1, in_channels=channel_lambda(160),
            out_channels=channel_lambda(320), stride=1, expansion_rate=expansion_rate
        )
        self.classifier = nn.Sequential(
            ConvBlock(
                in_channels=channel_lambda(320), out_channels=channel_lambda(1280), kernel_size=1,
                stride=1, padding=0, groups=1, activation=True
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=channel_lambda(1280), out_features=num_classes)
        )


    def forward(self, x):
        x = self.init_features(x)
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.classifier(x)

        return x

class BottleNeckLayer(nn.ModuleList):
    
    def __init__(self, num_blocks, in_channels, out_channels, stride, expansion_rate):
        super(BottleNeckLayer, self).__init__()

        downsample = True
        for _ in range(num_blocks):
            block = BottleNeckBlock(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, expansion_rate=expansion_rate, identity_downsample=downsample
            )
            in_channels = out_channels
            downsample = False
            stride = 1
            self.append(block)
        
    def forward(self, x):
        for block in self:
            x = block(x)

        return x

class BottleNeckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_rate, identity_downsample):
        super(BottleNeckBlock, self).__init__()

        self.expansion_pw = ConvBlock(
            in_channels=in_channels, out_channels=in_channels*expansion_rate,
            kernel_size=1, stride=1, padding=0, groups=1, activation=True
        )
        self.spatial_dw = ConvBlock(
            in_channels=in_channels*expansion_rate, out_channels=in_channels*expansion_rate,
            kernel_size=3, stride=stride, padding=1, groups=in_channels*expansion_rate, activation=True
        )
        self.transform_pw = ConvBlock(
            in_channels=in_channels*expansion_rate, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, groups=1, activation=False
        )

        if identity_downsample:
            self.identity_downsample = ConvBlock(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=stride, padding=0, groups=1, activation=False
            )
        else:
            self.identity_downsample = None

    def forward(self, x):

        out = self.expansion_pw(x)
        out = self.spatial_dw(out)
        out = self.transform_pw(out)

        x = self.identity_downsample(x) if self.identity_downsample is not None else x

        out = torch.add(x, out)
        out = nn.ReLU6(inplace=True)(out)

        return out

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activation):

        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU6(inplace=True) if activation else None

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
        if self.relu is not None:
            x = self.relu(x)
        
        return x
