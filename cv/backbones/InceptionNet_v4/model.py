import torch
import torch.nn as nn

from cv.utils import MetaWrapper

class Stem(nn.Module):

    def __init__(self, in_channels):

        super(Stem, self).__init__()

        self.step = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        self.branch1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch1_2 = ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0)

        self.branch2_1 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0)
        )
        self.branch2_2 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0)
        )

        self.branch3_1 = ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0)
        self.branch3_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.step(x)

        x_i = self.branch1_1(x)
        x_j = self.branch1_2(x)

        x = torch.cat([x_i, x_j], dim=1)

        x_i = self.branch2_1(x)
        x_j = self.branch2_2(x)

        x = torch.cat([x_i, x_j], dim=1)

        x_i = self.branch3_1(x)
        x_j = self.branch3_2(x)

        x = torch.cat([x_i, x_j], dim=1)

        return x

class InceptionA(nn.Module):

    def __init__(self):

        super(InceptionA, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=384, out_channels=96, kernel_size=1, stride=1, padding=0)
        )
        self.branch_2 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        )
        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)

        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)

        return x
    
class ReductionA(nn.Module):

    def __init__(self):
        
        super(ReductionA, self).__init__()

        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_2 = ConvBlock(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=0)
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=224, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=224, out_channels=256, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):

        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        return x

class InceptionB(nn.Module):

    def __init__(self):

        super(InceptionB, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0)
        )
        self.branch_2 = nn.Conv2d(in_channels=1024, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(in_channels=224, out_channels=224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )


    def forward(self, x):

        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)

        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)

        return x

class ReductionB(nn.Module):

    def __init__(self):

        super(ReductionB, self).__init__()

        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0)
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=256, out_channels=320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):

        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        return x
    
class InceptionC(nn.Module):

    def __init__(self):

        super(InceptionC, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0)
        )
        self.branch_2 = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.branch_3 = nn.Conv2d(in_channels=1536, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.branch_3_1 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch_3_2 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=384, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=384, out_channels=448, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels=448, out_channels=512, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.branch_4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch_4_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):

        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_3_1 = self.branch_3_1(x_3)
        x_3_2 = self.branch_3_2(x_3)
        x_4 = self.branch_4(x)
        x_4_1 = self.branch_4_1(x_4)
        x_4_2 = self.branch_4_2(x_4)

        x = torch.cat([x_1, x_2, x_3_1, x_3_2, x_4_1, x_4_2], dim=1)

        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1):

        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.init()

    def init(self):

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, val=0.0)

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Inceptionv4(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for InceptionNet-v4 architecture from paper on: Impact of Residual Connections on Learning"

    def __init__(self, blocks, in_channels=3, num_classes=1000, dropout=0.2):

        super(Inceptionv4, self).__init__()

        self.stem = Stem(in_channels=in_channels)
        self.inceptionA = nn.Sequential(*[InceptionA() for _ in range(blocks[0])])
        self.reductionA = ReductionA()
        self.inceptionB = nn.Sequential(*[InceptionB() for _ in range(blocks[1])])
        self.reductionB = ReductionB()
        self.inceptionC = nn.Sequential(*[InceptionC() for _ in range(blocks[2])])
        self.avrge_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1536, out_features=num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductionA(x)
        x = self.inceptionB(x)
        x = self.reductionB(x)
        x = self.inceptionC(x)
        x = self.avrge_pool(x)
        x = self.classifier(x)

        return x