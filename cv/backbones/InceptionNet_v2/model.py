import torch
import torch.nn as nn

from cv.utils import MetaWrapper


class Inceptionv2(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for InceptionNet-v2 architecture from paper on: Rethinking the Inception Architecture for Computer Vision"

    def __init__(self, num_classes, in_channels=3):
        super(Inceptionv2, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64,
            kernel_size=7, stride=2, padding=2
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2
        )
        self.conv2 = ConvBlock(
            in_channels=64, out_channels=80,
            kernel_size=3, stride=1, padding=0
        )
        self.conv3 = ConvBlock(
            in_channels=80, out_channels=192,
            kernel_size=3, stride=2, padding=0
        )
        self.conv4 = ConvBlock(
            in_channels=192, out_channels=288,
            kernel_size=3, stride=1, padding=1
        )
        self.inception_conv_factorized_1 = InceptionBlock_conv_factorized(
            in_channels=288, channels_1x1=96, channels_3x3_reduce=64,
            channels_3x3=96, channels_5x5_to_3x3_reduce=64,
            channels_5x5_to_3x3_1=64, channels_5x5_to_3x3_2=96, channels_pool_proj=96
        )
        self.inception_conv_factorized_2 = InceptionBlock_conv_factorized(
            in_channels=384, channels_1x1=96, channels_3x3_reduce=64,
            channels_3x3=96, channels_5x5_to_3x3_reduce=64,
            channels_5x5_to_3x3_1=64, channels_5x5_to_3x3_2=96, channels_pool_proj=96
        )
        self.inception_conv_factorized_3 = InceptionBlock_conv_factorized(
            in_channels=384, channels_1x1=96, channels_3x3_reduce=64,
            channels_3x3=96, channels_5x5_to_3x3_reduce=64,
            channels_5x5_to_3x3_1=64, channels_5x5_to_3x3_2=96, channels_pool_proj=96
        )
        self.inception_block_grid_reduction_1 = InceptionBlock_grid_reduction(
            in_channels=384, channels_1x1_branch_1=128, channels_3x3_1_branch_1=128,
            channels_3x3_2_branch_1=192, channels_1x1_branch_2=128, channels_3x3_branch_2=192
        )
        self.inception_block_asymmetric_conv_1 = InceptionBlock_asymmetric_conv(
            in_channels=768, channels_branch_1_1=160, channels_branch_1_2=160,
            channels_branch_1_3=160, channels_branch_1_4=160, channels_branch_1_5=160,
            channels_branch_2_1=160, channels_branch_2_2=160, channels_branch_2_3=160,
            channels_branch_3=160, channels_branch_4=160
        )
        self.inception_block_asymmetric_conv_2 = InceptionBlock_asymmetric_conv(
            in_channels=640, channels_branch_1_1=160, channels_branch_1_2=160,
            channels_branch_1_3=160, channels_branch_1_4=160, channels_branch_1_5=160,
            channels_branch_2_1=160, channels_branch_2_2=160, channels_branch_2_3=160,
            channels_branch_3=160, channels_branch_4=160
        )
        self.inception_block_asymmetric_conv_3 = InceptionBlock_asymmetric_conv(
            in_channels=640, channels_branch_1_1=160, channels_branch_1_2=160,
            channels_branch_1_3=160, channels_branch_1_4=160, channels_branch_1_5=160,
            channels_branch_2_1=160, channels_branch_2_2=160, channels_branch_2_3=160,
            channels_branch_3=160, channels_branch_4=160
        )
        self.inception_block_asymmetric_conv_4 = InceptionBlock_asymmetric_conv(
            in_channels=640, channels_branch_1_1=160, channels_branch_1_2=160,
            channels_branch_1_3=160, channels_branch_1_4=160, channels_branch_1_5=160,
            channels_branch_2_1=160, channels_branch_2_2=160, channels_branch_2_3=160,
            channels_branch_3=160, channels_branch_4=160
        )
        self.inception_block_asymmetric_conv_5 = InceptionBlock_asymmetric_conv(
            in_channels=640, channels_branch_1_1=160, channels_branch_1_2=160,
            channels_branch_1_3=160, channels_branch_1_4=160, channels_branch_1_5=160,
            channels_branch_2_1=160, channels_branch_2_2=160, channels_branch_2_3=160,
            channels_branch_3=160, channels_branch_4=160
        )
        self.inception_block_grid_reduction_2 = InceptionBlock_grid_reduction(
            in_channels=640, channels_1x1_branch_1=256, channels_3x3_1_branch_1=256,
            channels_3x3_2_branch_1=320, channels_1x1_branch_2=256, channels_3x3_branch_2=320
        )
        self.inception_block_expanded_filter_bank_1 = InceptionBlock_expanded_filter_bank(
            in_channels=1280, channels_branch_1_1=192, channels_branch_1_2=192,
            channels_branch_1_sub_1=256, channels_branch_1_sub_2=256, channels_branch_2_1=128,
            channels_branch_2_sub_1=192, channels_branch_2_sub_2=192, channels_branch_3=64, channels_branch_4=64
        )
        self.inception_block_expanded_filter_bank_2 = InceptionBlock_expanded_filter_bank(
            in_channels=1024, channels_branch_1_1=256, channels_branch_1_2=256,
            channels_branch_1_sub_1=384, channels_branch_1_sub_2=384, channels_branch_2_1=256,
            channels_branch_2_sub_1=384, channels_branch_2_sub_2=384, channels_branch_3=256, channels_branch_4=256
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Dropout(p=0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, num_classes)
        )

        self.aux_classifier_1 = AuxiliaryClassifier(
            conv_in_channels=768, conv_out_channels=128, num_classes=num_classes
        )

        self.getLayersToCUDA()

    def getLayersToCUDA(self):
        if GPU_Support.support_gpu == 1:
            self.conv1.to(device="cuda:0")
            self.maxpool1.to(device="cuda:0")
            self.conv2.to(device="cuda:0")
            self.conv3.to(device="cuda:0")
            self.conv4.to(device="cuda:0")
            self.inception_conv_factorized_1.to(device="cuda:0")
            self.inception_conv_factorized_2.to(device="cuda:0")
            self.inception_conv_factorized_3.to(device="cuda:0")
            self.inception_block_grid_reduction_1.to(device="cuda:0")
            self.aux_classifier_1.to(device="cuda:0")
            self.inception_block_asymmetric_conv_1.to(device="cuda:0")
            self.inception_block_asymmetric_conv_2.to(device="cuda:0")
            self.inception_block_asymmetric_conv_3.to(device="cuda:0")
            self.inception_block_asymmetric_conv_4.to(device="cuda:0")
            self.inception_block_asymmetric_conv_5.to(device="cuda:0")
            self.inception_block_grid_reduction_2.to(device="cuda:0")
            self.inception_block_expanded_filter_bank_1.to(device="cuda:0")
            self.inception_block_expanded_filter_bank_2.to(device="cuda:0")
            self.classifier.to(device="cuda:0")

        
    def forward(self, x, phase="validation"):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.inception_conv_factorized_1(x)
        x = self.inception_conv_factorized_2(x)
        x = self.inception_conv_factorized_3(x)
        x = self.inception_block_grid_reduction_1(x)

        if phase == "training":
            aux_1_out = self.aux_classifier_1(x)

        x = self.inception_block_asymmetric_conv_1(x)
        x = self.inception_block_asymmetric_conv_2(x)
        x = self.inception_block_asymmetric_conv_3(x)
        x = self.inception_block_asymmetric_conv_4(x)
        x = self.inception_block_asymmetric_conv_5(x)
        x = self.inception_block_grid_reduction_2(x)
            
        x = self.inception_block_expanded_filter_bank_1(x)
        x = self.inception_block_expanded_filter_bank_2(x)
        x = self.classifier(x)

        if phase == "training":
            return [aux_1_out, x]
        else:
            return x

class InceptionBlock_grid_reduction(nn.Module):

    def __init__(self, in_channels, channels_1x1_branch_1, channels_3x3_1_branch_1,
                channels_3x3_2_branch_1, channels_1x1_branch_2, channels_3x3_branch_2):
        
        super(InceptionBlock_grid_reduction, self).__init__()

        self.branch_1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_branch_1,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_1x1_branch_1, out_channels=channels_3x3_1_branch_1,
                kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=channels_3x3_1_branch_1, out_channels=channels_3x3_2_branch_1,
                kernel_size=3, stride=2, padding=0
            )
        )
        self.branch_2 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1_branch_2,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_1x1_branch_2, out_channels=channels_3x3_branch_2,
                kernel_size=3, stride=2, padding=0
            )
        )
        self.branch_3 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        branch_3_out = self.branch_3(x)

        out = torch.cat([branch_1_out, branch_2_out, branch_3_out], dim=1)

        return out

class InceptionBlock_asymmetric_conv(nn.Module):

    def __init__(self, in_channels, channels_branch_1_1, channels_branch_1_2,
                channels_branch_1_3, channels_branch_1_4, channels_branch_1_5,
                channels_branch_2_1, channels_branch_2_2, channels_branch_2_3,
                channels_branch_3, channels_branch_4, n=7):
        
        super(InceptionBlock_asymmetric_conv, self).__init__()

        self.branch_1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_branch_1_1,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_branch_1_1, out_channels=channels_branch_1_2,
                kernel_size=(1, n), stride=1, padding=(0, n // 2)
            ),
            ConvBlock(
                in_channels=channels_branch_1_2, out_channels=channels_branch_1_3,
                kernel_size=(n, 1), stride=1, padding=(n // 2, 0)
            ),
            ConvBlock(
                in_channels=channels_branch_1_3, out_channels=channels_branch_1_4,
                kernel_size=(1, n), stride=1, padding=(0, n // 2)
            ),
            ConvBlock(
                in_channels=channels_branch_1_4, out_channels=channels_branch_1_5,
                kernel_size=(n, 1), stride=1, padding=(n // 2, 0)
            )
        )

        self.branch_2 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_branch_2_1,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_branch_2_1, out_channels=channels_branch_2_2,
                kernel_size=(1, n), stride=1, padding=(0, n // 2)
            ),
            ConvBlock(
                in_channels=channels_branch_2_2, out_channels=channels_branch_2_3,
                kernel_size=(n, 1), stride=1, padding=(n // 2, 0)
            )
        )

        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=in_channels, out_channels=channels_branch_3,
                kernel_size=1, stride=1, padding=0
            )
        )

        self.branch_4 = ConvBlock(
            in_channels=in_channels, out_channels=channels_branch_4,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        branch_3_out = self.branch_3(x)
        branch_4_out = self.branch_4(x)

        out = torch.cat([branch_1_out, branch_2_out, branch_3_out, branch_4_out], dim=1)

        return out

class InceptionBlock_conv_factorized(nn.Module):

    def __init__(self, in_channels, channels_1x1, channels_3x3_reduce,
                channels_3x3, channels_5x5_to_3x3_reduce, channels_5x5_to_3x3_1,
                channels_5x5_to_3x3_2, channels_pool_proj):
        
        super(InceptionBlock_conv_factorized, self).__init__()

        self.branch_1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_1x1,
                kernel_size=1, stride=1, padding=0
            )
        )
        self.branch_2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=in_channels, out_channels=channels_pool_proj,
                kernel_size=1, stride=1, padding=0
            )
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_3x3_reduce,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_3x3_reduce, out_channels=channels_3x3,
                kernel_size=3, stride=1, padding=1
            )
        )
        self.branch_4 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_5x5_to_3x3_reduce,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_5x5_to_3x3_reduce, out_channels=channels_5x5_to_3x3_1,
                kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=channels_5x5_to_3x3_1, out_channels=channels_5x5_to_3x3_2,
                kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        branch_3_out = self.branch_3(x)
        branch_4_out = self.branch_4(x)

        return torch.cat([branch_1_out, branch_2_out, branch_3_out, branch_4_out], dim=1)

class InceptionBlock_expanded_filter_bank(nn.Module):

    def __init__(self, in_channels, channels_branch_1_1, channels_branch_1_2,
                channels_branch_1_sub_1, channels_branch_1_sub_2, channels_branch_2_1,
                channels_branch_2_sub_1, channels_branch_2_sub_2, channels_branch_3, channels_branch_4):
        
        super(InceptionBlock_expanded_filter_bank, self).__init__()

        self.branch_1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_branch_1_1,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_branch_1_1, out_channels=channels_branch_1_2,
                kernel_size=3, stride=1, padding=1
            )
        )

        self.branch_1_sub_1 = ConvBlock(
            in_channels=channels_branch_1_2, out_channels=channels_branch_1_sub_1,
            kernel_size=(1, 3), stride=1, padding=(0, 3 // 2)
        )
        self.branch_1_sub_2 = ConvBlock(
            in_channels=channels_branch_1_2, out_channels=channels_branch_1_sub_2,
            kernel_size=(3, 1), stride=1, padding=(3 // 2, 0)
        )

        self.branch_2 = ConvBlock(
            in_channels=in_channels, out_channels=channels_branch_2_1,
            kernel_size=1, stride=1, padding=0
        )
        self.branch_2_sub_1 = ConvBlock(
            in_channels=channels_branch_2_1, out_channels=channels_branch_2_sub_1,
            kernel_size=(1, 3), stride=1, padding=(0, 3 // 2)
        )
        self.branch_2_sub_2 = ConvBlock(
            in_channels=channels_branch_2_1, out_channels=channels_branch_2_sub_2,
            kernel_size=(3, 1), stride=1, padding=(3 // 2, 0)
        )
        
        self.branch_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(
                in_channels=in_channels, out_channels=channels_branch_3,
                kernel_size=1, stride=1, padding=0
            )
        )

        self.branch_4 = ConvBlock(
            in_channels=in_channels, out_channels=channels_branch_4,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        branch_1_out = self.branch_1(x)
        branch_1_sub_1_out = self.branch_1_sub_1(branch_1_out)
        branch_1_sub_2_out = self.branch_1_sub_2(branch_1_out)

        branch_2_out = self.branch_2(x)
        branch_2_sub_1_out = self.branch_2_sub_1(branch_2_out)
        branch_2_sub_2_out = self.branch_2_sub_2(branch_2_out)

        branch_3_out = self.branch_3(x)
        branch_4_out = self.branch_4(x)

        out = torch.cat([
            branch_1_sub_1_out, branch_1_sub_2_out, branch_2_sub_1_out, branch_2_sub_2_out, branch_3_out, branch_4_out
        ], dim=1)
        
        return out

class AuxiliaryClassifier(nn.Module):
        def __init__(self, conv_in_channels, conv_out_channels, num_classes):
            super(AuxiliaryClassifier, self).__init__()

            self.aux_classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                ConvBlock(
                    in_channels=conv_in_channels, out_channels=conv_out_channels,
                    kernel_size=1, stride=1, padding=0
                ),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=5*5*conv_out_channels, out_features=1024),
                nn.ReLU(),
                nn.Dropout(p=0.7),
                nn.Linear(in_features=1024, out_features=num_classes)
            )

        def forward(self, x):
            x = self.aux_classifier(x)

            return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, val=0.0)

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

