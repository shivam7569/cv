import numpy as np
import torch
import torch.nn as nn

from src.gpu_devices import GPU_Support


class ResidualGroup(nn.ModuleList):
    def __init__(self, num_blocks, in_channels, channels_1x1_in, channels_3x3,
                channels_1x1_out, bold=True, reduction_ratio=16):
        super(ResidualGroup, self).__init__()

        downsample = True
        for _ in range(num_blocks):
            block = ResidualBlock(
                in_channels=in_channels, channels_1x1_in=channels_1x1_in, channels_3x3=channels_3x3,
                channels_1x1_out=channels_1x1_out, bold=bold, downsample=downsample, r=reduction_ratio
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
                channels_1x1_out, bold=True, downsample=True, r=16):
        
        super(ResidualBlock, self).__init__()

        downsample_conv_stride = 1 if bold else 2
        self.block = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_in,
                kernel_size=1, stride=1, padding=0, activation=True
            ),
            ConvBlock(
                in_channels=channels_1x1_in, out_channels=channels_3x3,
                kernel_size=3, stride=downsample_conv_stride, padding=1, activation=True
            ),
            ConvBlock(
                in_channels=channels_3x3, out_channels=channels_1x1_out,
                kernel_size=1, stride=1, padding=0, activation=False
            )
        )

        if downsample:
            downsample_stride = 1 if bold else 2
            self.identity_downsample = ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_out,
                kernel_size=1, stride=downsample_stride, padding=0, activation=False
            )
            self.initializeConv(kernel_size=1, channels=channels_1x1_out)
        else:
            self.identity_downsample = None

        self.se_block = SEBlock(
            in_channels=channels_1x1_out, r=r
        )

    def initializeConv(self, kernel_size, channels):
        init_n = (kernel_size ** 2) * channels
        init_mean = 0.0
        init_std = np.sqrt(2 / init_n)
        
        nn.init.normal_(self.identity_downsample.conv.weight, mean=init_mean, std=init_std)
        nn.init.constant_(self.identity_downsample.conv.bias, val=0.0)

    def forward(self, x):
        block_out = self.block(x)
        seBlock_out = self.se_block(block_out).unsqueeze(-1).unsqueeze(-1)

        residual_se_rescale = torch.mul(block_out, seBlock_out)

        if self.identity_downsample is not None:    
            x = self.identity_downsample(x)

        out = torch.add(x, residual_se_rescale)
        out = nn.ReLU(inplace=True)(out)

        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
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

class SEBlock(nn.Module):

    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=int(in_channels/r)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=int(in_channels/r), out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = self.excitation(x)

        return x

class SENet(nn.Module):

    def __init__(self, num_classes, in_channels=3, reduction_ratio=16):
        super(SENet, self).__init__()

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
            num_blocks=3, in_channels=64, channels_1x1_in=64,
            channels_3x3=64, channels_1x1_out=256, bold=True, reduction_ratio=reduction_ratio
        )
        self.res_group_2 = ResidualGroup(
            num_blocks=4, in_channels=256, channels_1x1_in=128,
            channels_3x3=128, channels_1x1_out=512, bold=False, reduction_ratio=reduction_ratio
        )
        self.res_group_3 = ResidualGroup(
            num_blocks=6, in_channels=512, channels_1x1_in=256,
            channels_3x3=256, channels_1x1_out=1024, bold=False, reduction_ratio=reduction_ratio
        )
        self.res_group_4 = ResidualGroup(
            num_blocks=3, in_channels=1024, channels_1x1_in=512,
            channels_3x3=512, channels_1x1_out=2048, bold=False, reduction_ratio=reduction_ratio
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

        self.initializeConv()
        self.getLayersToCuda()

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

    def getLayersToCuda(self):
        if GPU_Support.support_gpu == 2:
            self.init_features.to("cuda:0")
            self.res_group_1.to("cuda:0")
            self.res_group_2.to("cuda:0")

            self.res_group_3.to("cuda:1")
            self.res_group_4.to("cuda:1")
            self.classifier.to("cuda:1")
        elif GPU_Support.support_gpu == 1:
            self.init_features.to("cuda:0")
            self.res_group_1.to("cuda:0")
            self.res_group_2.to("cuda:0")
            self.res_group_3.to("cuda:0")
            self.res_group_4.to("cuda:0")
            self.classifier.to("cuda:0")

    def forward(self, x):
        x = self.init_features(x)
        x = self.res_group_1(x)
        x = self.res_group_2(x)
        x = self.res_group_3(x)
        x = self.res_group_4(x)
        x = self.classifier(x)

        return x
