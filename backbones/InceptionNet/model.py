import numpy as np
import torch
import torch.nn as nn

from src.gpu_devices import GPU_Support

"""
The code has nested classes, which I later realised is not a good practice
"""

class Inception(nn.Module):

    class InceptioModule(nn.Module):
        def __init__(self, in_channels,
                    channels_1x1, channels_3x3_reduce,
                    channels_3x3, channels_5x5_reduce,
                    channels_5x5, channels_pool_proj):
            
            super(Inception.InceptioModule, self).__init__()

            self.branch_1 = Inception.ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1,
                kernel_size=1, stride=1, padding=0
            )

            self.branch_2 = nn.Sequential(
                Inception.ConvBlock(
                    in_channels=in_channels, out_channels=channels_3x3_reduce,
                    kernel_size=1, stride=1, padding=0
                ),
                Inception.ConvBlock(
                    in_channels=channels_3x3_reduce, out_channels=channels_3x3,
                    kernel_size=3, stride=1, padding=1
                )
            )

            self.branch_3 = nn.Sequential(
                Inception.ConvBlock(
                    in_channels=in_channels, out_channels=channels_5x5_reduce,
                    kernel_size=1, stride=1, padding=0
                ),
                Inception.ConvBlock(
                    in_channels=channels_5x5_reduce, out_channels=channels_5x5,
                    kernel_size=5, stride=1, padding=2
                )
            )

            self.branch_4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                Inception.ConvBlock(
                    in_channels=in_channels, out_channels=channels_pool_proj,
                    kernel_size=1, stride=1, padding=0
                )
            )

        def forward(self, x):
            branch_1_out = self.branch_1(x)
            branch_2_out = self.branch_2(x)
            branch_3_out = self.branch_3(x)
            branch_4_out = self.branch_4(x)

            out = torch.cat(
                [branch_1_out, branch_2_out, branch_3_out, branch_4_out], dim=1
            )

            return out

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels,
                    kernel_size, stride, padding):
            
            super(Inception.ConvBlock, self).__init__()

            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.bn = nn.BatchNorm2d(num_features=out_channels)
            self.relu = nn.ReLU()

            self.initializeConv()

        def initializeConv(self):
            nn.init.xavier_normal_(self.conv.weight)
            nn.init.constant_(self.conv.bias, val=0.0)

            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

            return x

    class AuxiliaryClassifier(nn.Module):
        def __init__(self, conv_in_channels, conv_out_channels, num_classes):
            super(Inception.AuxiliaryClassifier, self).__init__()

            self.aux_classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                Inception.ConvBlock(
                    in_channels=conv_in_channels, out_channels=conv_out_channels,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=4*4*conv_out_channels, out_features=1024),
                nn.ReLU(),
                nn.Dropout(p=0.7),
                nn.Linear(in_features=1024, out_features=num_classes)
            )

        def forward(self, x):
            x = self.aux_classifier(x)

            return x
    
    def __init__(self, num_classes, in_channels=3):
        super(Inception, self).__init__()

        self.inception = nn.Sequential(
            Inception.ConvBlock(
                in_channels=in_channels, out_channels=64,
                kernel_size=7, stride=2, padding=3
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception.ConvBlock(
                in_channels=64, out_channels=64,
                kernel_size=1, stride=1, padding=0
            ),
            Inception.ConvBlock(
                in_channels=64, out_channels=192,
                kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception.InceptioModule(
                in_channels=192, channels_1x1=64, channels_3x3_reduce=96,
                channels_3x3=128, channels_5x5_reduce=16,
                channels_5x5=32, channels_pool_proj=32
            ),
            Inception.InceptioModule(
                in_channels=256, channels_1x1=128, channels_3x3_reduce=128,
                channels_3x3=192, channels_5x5_reduce=32,
                channels_5x5=96, channels_pool_proj=64
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception.InceptioModule(
                in_channels=480, channels_1x1=192, channels_3x3_reduce=96,
                channels_3x3=208, channels_5x5_reduce=16,
                channels_5x5=48, channels_pool_proj=64
            ),
            Inception.InceptioModule(
                in_channels=512, channels_1x1=160, channels_3x3_reduce=112,
                channels_3x3=224, channels_5x5_reduce=24,
                channels_5x5=64, channels_pool_proj=64
            ),
            Inception.InceptioModule(
                in_channels=512, channels_1x1=128, channels_3x3_reduce=128,
                channels_3x3=256, channels_5x5_reduce=24,
                channels_5x5=64, channels_pool_proj=64
            ),
            Inception.InceptioModule(
                in_channels=512, channels_1x1=112, channels_3x3_reduce=144,
                channels_3x3=288, channels_5x5_reduce=32,
                channels_5x5=64, channels_pool_proj=64
            ),
            Inception.InceptioModule(
                in_channels=528, channels_1x1=256, channels_3x3_reduce=160,
                channels_3x3=320, channels_5x5_reduce=32,
                channels_5x5=128, channels_pool_proj=128
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception.InceptioModule(
                in_channels=832, channels_1x1=256, channels_3x3_reduce=160,
                channels_3x3=320, channels_5x5_reduce=32,
                channels_5x5=128, channels_pool_proj=128
            ),
            Inception.InceptioModule(
                in_channels=832, channels_1x1=384, channels_3x3_reduce=192,
                channels_3x3=384, channels_5x5_reduce=48,
                channels_5x5=128, channels_pool_proj=128
            ),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Dropout(p=0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        self.inception_block_1 = self.inception[:9]
        self.inception_block_2 = self.inception[9: 12]
        self.inception_block_3 = self.inception[12:]

        self.aux_classifier_1 = Inception.AuxiliaryClassifier(
            conv_in_channels=512, conv_out_channels=128, num_classes=num_classes
        )
        self.aux_classifier_2 = Inception.AuxiliaryClassifier(
            conv_in_channels=528, conv_out_channels=128, num_classes=num_classes
        )

        if GPU_Support.support_gpu == 2:
            self.inception_block_1.to("cuda:0")
            self.inception_block_2.to("cuda:0")
            self.inception_block_3.to("cuda:1")

            self.aux_classifier_1.to("cuda:1")
            self.aux_classifier_2.to("cuda:1")
        elif GPU_Support.support_gpu == 1:
            self.inception_block_1.to("cuda:0")
            self.inception_block_2.to("cuda:0")
            self.inception_block_3.to("cuda:0")

            self.aux_classifier_1.to("cuda:0")
            self.aux_classifier_2.to("cuda:0")

    @staticmethod
    def get_layer_out(layer, input_):
        if GPU_Support.support_gpu > 1:
            layer_device = next(layer.parameters()).device
            input_ = input_.cpu().to(layer_device)
            out = layer(input_)
        else:
            out = layer(input_)
        return out

    def forward(self, x, phase="validation"):
        if phase == "training":
            inception_1_block_out = Inception.get_layer_out(layer=self.inception_block_1, input_=x)
            inception_2_block_out = Inception.get_layer_out(layer=self.inception_block_2, input_=inception_1_block_out)
            inception_out = Inception.get_layer_out(layer=self.inception_block_3, input_=inception_2_block_out)

            aux_classifier_1_out = Inception.get_layer_out(layer=self.aux_classifier_1, input_=inception_1_block_out)
            aux_classifier_2_out = Inception.get_layer_out(layer=self.aux_classifier_2, input_=inception_2_block_out)

            return [aux_classifier_1_out, aux_classifier_2_out, inception_out]
    
        else:
            inception_1_block_out = Inception.get_layer_out(layer=self.inception_block_1, input_=x)
            inception_2_block_out = Inception.get_layer_out(layer=self.inception_block_2, input_=inception_1_block_out)
            inception_out = Inception.get_layer_out(layer=self.inception_block_3, input_=inception_2_block_out)

            return inception_out
