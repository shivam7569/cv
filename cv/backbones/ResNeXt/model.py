import numpy as np
import torch
import torch.nn as nn

from cv.utils import MetaWrapper


class ResidualGroup(nn.ModuleList):
    def __init__(self, num_blocks, in_channels, channels_1x1_in,
                channels_3x3, channels_1x1_out, bold=True, cardinality=32):
        
        super(ResidualGroup, self).__init__()

        downsample = True
        for _ in range(num_blocks):
            block = ResidualBlock(
                in_channels=in_channels, channels_1x1_in=channels_1x1_in,
                channels_3x3=channels_3x3, channels_1x1_out=channels_1x1_out,
                bold=bold, cardinality=cardinality, downsample=downsample
            )
            in_channels = channels_1x1_out
            if not bold: bold = True
            downsample = False
            self.append(block)

    def forward(self, x):
        for block in self:
            x = block(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels_1x1_in, channels_3x3,
                channels_1x1_out, bold=True, cardinality=32, downsample=True):
        
        super(ResidualBlock, self).__init__()

        downsample_conv_stride = 1 if bold else 2

        self.block = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_in,
                kernel_size=1, stride=1, padding=0,
                groups=1, activation=True
            ),
            ConvBlock(
                in_channels=channels_1x1_in, out_channels=channels_3x3,
                kernel_size=3, stride=downsample_conv_stride, padding=1,
                groups=cardinality, activation=True
            ),
            ConvBlock(
                in_channels=channels_3x3, out_channels=channels_1x1_out,
                kernel_size=1, stride=1, padding=0,
                groups=1, activation=False
            )
        )

        if downsample:
            identity_downsample_stride = 1 if bold else 2
            self.identity_downsample = ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_out,
                kernel_size=1, stride=identity_downsample_stride, padding=0, groups=1, activation=False
            )
            self.initializeConv(kernel_size=1, channels=channels_1x1_out)
        else:
            self.identity_downsample = None

    def initializeConv(self, kernel_size, channels):
        init_n = (kernel_size ** 2) * channels
        init_mean = 0.0
        init_std = np.sqrt(2 / init_n)
        
        nn.init.normal_(self.identity_downsample.conv.weight, mean=init_mean, std=init_std)
        nn.init.constant_(self.identity_downsample.conv.bias, val=0.0)

    def forward(self, x):
        block_out = self.block(x)
        if self.identity_downsample is not None:
            x = self.identity_downsample(x)
        out = torch.add(x, block_out)
        out = nn.ReLU(inplace=True)(out)

        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activation=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True) if activation else None

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
        x = self.relu(x) if self.relu is not None else x

        return x

class ResNeXt(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for ResNeXt architecture from paper on: Aggregated Residual Transformations for Deep Neural Networks"

    def __init__(self, num_classes, in_channels=3):
        super(ResNeXt, self).__init__()

        self.init_features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=64,
                kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res_group_1 = ResidualGroup(
            num_blocks=3, in_channels=64, channels_1x1_in=128,
            channels_3x3=128, channels_1x1_out=256, bold=True, cardinality=32
        )
        self.res_group_2 = ResidualGroup(
            num_blocks=4, in_channels=256, channels_1x1_in=256,
            channels_3x3=256, channels_1x1_out=512, bold=False, cardinality=32
        )
        self.res_group_3 = ResidualGroup(
            num_blocks=6, in_channels=512, channels_1x1_in=512,
            channels_3x3=512, channels_1x1_out=1024, bold=False, cardinality=32
        )
        self.res_group_4 = ResidualGroup(
            num_blocks=3, in_channels=1024, channels_1x1_in=1024,
            channels_3x3=1024, channels_1x1_out=2048, bold=False, cardinality=32
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

        self.initializeConv()

    def initializeConv(self):
        for module in self.init_features.modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                bias = module.bias

                init_n = (module.kernel_size[0] ** 2) * module.out_channels
                init_mean = 0.0
                init_std = np.sqrt(2 / init_n)

                nn.init.normal_(weight, mean=init_mean, std=init_std)
                nn.init.constant_(bias, val=0.0)

    def forward(self, x):
        x = self.init_features(x)
        x = self.res_group_1(x)
        x = self.res_group_2(x)
        x = self.res_group_3(x)
        x = self.res_group_4(x)
        x = self.classifier(x)

        return x
