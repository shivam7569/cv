import torch
import torch.nn as nn

from cv.utils import MetaWrapper

"""
The code has nested classes, which I later realised is not a good practice
"""

class InceptioModule(nn.Module):

    def __init__(self, in_channels,
                channels_1x1, channels_3x3_reduce,
                channels_3x3, channels_5x5_reduce,
                channels_5x5, channels_pool_proj):
        
        super(InceptioModule, self).__init__()

        self.branch_1 = ConvBlock(
            in_channels=in_channels, out_channels=channels_1x1,
            kernel_size=1, stride=1, padding=0
        )

        self.branch_2 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_3x3_reduce,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_3x3_reduce, out_channels=channels_3x3,
                kernel_size=3, stride=1, padding=1
            )
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=channels_5x5_reduce,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=channels_5x5_reduce, out_channels=channels_5x5,
                kernel_size=5, stride=1, padding=2
            )
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(
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
        
        super(ConvBlock, self).__init__()

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
        super(AuxiliaryClassifier, self).__init__()

        self.aux_classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(
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
        

class Inception(nn.Module, metaclass=MetaWrapper):

    """
    Inception-v1 model class based on the architecture proposed in the 

    This class implements the Inception architecture, including several Inception modules and 
    auxiliary classifiers to improve training performance. It processes input data through
    multiple branches of convolutional layers, where each branch has a different kernel size to 
    capture features at various receptive fields.

    Args:
        num_classes (int): Number of output classes.
        in_channels (int, optional): Number of input channels, typically 3 for RGB (default: 3).

    Example:
        >>> model = Inception(num_classes=1000)
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for Inception-v1 architecture from paper on: Going deeper with convolutions"
    
    def __init__(self, num_classes, in_channels=3):
        super(Inception, self).__init__()

        self.inception = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, out_channels=64,
                kernel_size=7, stride=2, padding=3
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(
                in_channels=64, out_channels=64,
                kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=64, out_channels=192,
                kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptioModule(
                in_channels=192, channels_1x1=64, channels_3x3_reduce=96,
                channels_3x3=128, channels_5x5_reduce=16,
                channels_5x5=32, channels_pool_proj=32
            ),
            InceptioModule(
                in_channels=256, channels_1x1=128, channels_3x3_reduce=128,
                channels_3x3=192, channels_5x5_reduce=32,
                channels_5x5=96, channels_pool_proj=64
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptioModule(
                in_channels=480, channels_1x1=192, channels_3x3_reduce=96,
                channels_3x3=208, channels_5x5_reduce=16,
                channels_5x5=48, channels_pool_proj=64
            ),
            InceptioModule(
                in_channels=512, channels_1x1=160, channels_3x3_reduce=112,
                channels_3x3=224, channels_5x5_reduce=24,
                channels_5x5=64, channels_pool_proj=64
            ),
            InceptioModule(
                in_channels=512, channels_1x1=128, channels_3x3_reduce=128,
                channels_3x3=256, channels_5x5_reduce=24,
                channels_5x5=64, channels_pool_proj=64
            ),
            InceptioModule(
                in_channels=512, channels_1x1=112, channels_3x3_reduce=144,
                channels_3x3=288, channels_5x5_reduce=32,
                channels_5x5=64, channels_pool_proj=64
            ),
            InceptioModule(
                in_channels=528, channels_1x1=256, channels_3x3_reduce=160,
                channels_3x3=320, channels_5x5_reduce=32,
                channels_5x5=128, channels_pool_proj=128
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptioModule(
                in_channels=832, channels_1x1=256, channels_3x3_reduce=160,
                channels_3x3=320, channels_5x5_reduce=32,
                channels_5x5=128, channels_pool_proj=128
            ),
            InceptioModule(
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

        self.aux_classifier_1 = AuxiliaryClassifier(
            conv_in_channels=512, conv_out_channels=128, num_classes=num_classes
        )
        self.aux_classifier_2 = AuxiliaryClassifier(
            conv_in_channels=528, conv_out_channels=128, num_classes=num_classes
        )

    def forward(self, x, phase="validation"):
        """
        Forward pass through the Inception model.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, in_channels, height, width)`.
            phase (str, optional): Phase of model training ('training' or 'validation') (default: "validation").

        Returns:
            Tensor or list: In validation phase, returns the final output tensor.
            In training phase, returns a list containing outputs from auxiliary classifiers and the final output.

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        if phase == "training":
            inception_1_block_out = self.inception_block_1(x)
            inception_2_block_out = self.inception_block_2(inception_1_block_out)
            inception_out = self.inception_block_3(inception_2_block_out)

            aux_classifier_1_out = self.aux_classifier_1(inception_1_block_out)
            aux_classifier_2_out = self.aux_classifier_2(inception_2_block_out)

            return [aux_classifier_1_out, aux_classifier_2_out, inception_out]
    
        else:
            inception_1_block_out = self.inception_block_1(x)
            inception_2_block_out = self.inception_block_2(inception_1_block_out)
            inception_out = self.inception_block_3(inception_2_block_out)

            return inception_out
