import torch
import numpy as np
import torch.nn as nn

from cv.utils import MetaWrapper
from cv.attention.variants import BottleNeck_MHSA

class BoT_ViTParams:

    MHSA_NUM_HEADS = None
    MHSA_DROPOUT = 0.0

class ResidualGroup(nn.ModuleList):

    def __init__(self, num_blocks, in_channels, channels_1x1_in, channels_3x3, channels_1x1_out, bold=True, last_group=False):
        super(ResidualGroup, self).__init__()

        downsample = True
        for _ in range(num_blocks):
            block = ResidualBlock(
                in_channels=in_channels, channels_1x1_in=channels_1x1_in, channels_3x3=channels_3x3,
                channels_1x1_out=channels_1x1_out, bold=bold, downsample=downsample, last_group=last_group
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
                channels_1x1_out, bold=True, downsample=True, last_group=False):
        
        super(ResidualBlock, self).__init__()

        downsample_conv_stride = 1 if bold else 2

        if not last_group:
            self.block = nn.ModuleList(
                [
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
                ]
            )
        else:

            downsample_layer = nn.Identity() if bold else nn.AvgPool2d(kernel_size=2, stride=2)

            self.block = nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=in_channels, out_channels=channels_1x1_in,
                        kernel_size=1, stride=1, padding=0, activation=True
                    ),
                    BottleNeck_MHSA(
                        embed_dims=channels_1x1_in, num_heads=BoT_ViTParams.MHSA_NUM_HEADS, feature_map_size=(64 if not bold else 32, 64 if not bold else 32),
                        attention_dropout=BoT_ViTParams.MHSA_DROPOUT, qkv_bias=False
                    ),
                    downsample_layer,
                    ConvBlock(
                        in_channels=channels_3x3, out_channels=channels_1x1_out,
                        kernel_size=1, stride=1, padding=0, activation=False
                    )
                ]
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

    def initializeConv(self, kernel_size, channels):
        init_n = (kernel_size ** 2) * channels
        init_mean = 0.0
        init_std = np.sqrt(2 / init_n)
        
        nn.init.normal_(self.identity_downsample.conv.weight, mean=init_mean, std=init_std)
        nn.init.constant_(self.identity_downsample.conv.bias, val=0.0)

    def forward(self, x):
        res_x = x.clone()
        for blk in self.block:
            x = blk(x)
            
        if self.identity_downsample is not None:    
            res_x = self.identity_downsample(res_x)
        out = torch.add(res_x, x)
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

class BoT_ViT(nn.Module, metaclass=MetaWrapper):

    """
    `BoT-ViT` (Bottleneck Transformers for Visual Recognition) model architecture from `paper <https://arxiv.org/abs/2101.11605.pdf>`_.

    This class implements the BoT-ViT model, combining convolutional and bottleneck transformer blocks for 
    feature extraction and classification. The model processes input images through multiple residual groups 
    and a final classification layer.

    Args:
        mhsa_num_heads (int): The number of heads for the multi-head self-attention layer.
        attention_dropout (float, optional): Dropout rate for attention layers (default: 0.0).
        num_classes (int, optional): Number of output classes for classification (default: 1000).
        in_channels (int, optional): Number of input channels for the images, typically 3 for RGB (default: 3).

    Example:
        >>> model = BoT_ViT(mhsa_num_heads=8, attention_dropout=0.1, num_classes=1000)
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for BoT-ViT architecture from paper on: Bottleneck Transformers for Visual Recognition"

    def __init__(self, mhsa_num_heads, attention_dropout=0.0, num_classes=1000, in_channels=3):

        super(BoT_ViT, self).__init__()

        BoT_ViTParams.MHSA_NUM_HEADS = mhsa_num_heads
        BoT_ViTParams.MHSA_DROPOUT = attention_dropout

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
            channels_3x3=64, channels_1x1_out=256, bold=True
        )
        self.res_group_2 = ResidualGroup(
            num_blocks=4, in_channels=256, channels_1x1_in=128,
            channels_3x3=128, channels_1x1_out=512, bold=False
        )
        self.res_group_3 = ResidualGroup(
            num_blocks=6, in_channels=512, channels_1x1_in=256,
            channels_3x3=256, channels_1x1_out=1024, bold=False
        )
        self.res_group_4 = ResidualGroup(
            num_blocks=3, in_channels=1024, channels_1x1_in=512,
            channels_3x3=512, channels_1x1_out=2048, bold=False, last_group=True
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

        self.initializeConv()

    def initializeConv(self):

        """
        Initialize convolutional layers with specific weight and bias initialization.

        This method initializes the weights and biases of convolutional layers using a specific strategy:
        - **Weights**: Initialized with a normal distribution where the mean is `0.0` and the standard deviation
        is computed based on the number of input units and output channels.
        - **Biases**: Initialized to `0.0`.

        The standard deviation for weight initialization is calculated using the formula:

        .. math::

            \\text{std} = \\sqrt{\\frac{2}{n_{\\text{in}}}}

        where:

        .. math::

            n_{\\text{in}} = \\text{kernel_size[0]}^2 \\times \\text{out_channels}

        The weight initialization is performed as follows:

        .. math::

            \\text{weight} \\sim \\mathcal{N}(\\text{mean}=0.0, \\text{std})

        Biases are initialized with:

        .. math::

            \\text{bias} = 0.0

        This initialization helps in stabilizing the learning process and improving the convergence rate.
        """

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
        """
        Defines the forward pass through the BoT-ViT model.

        The input tensor passes through the initial convolutional layers, followed by multiple residual groups, 
        and is then processed through the classifier for final classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the model, with shape (batch_size, num_classes).

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        x = self.init_features(x)
        x = self.res_group_1(x)
        x = self.res_group_2(x)
        x = self.res_group_3(x)
        x = self.res_group_4(x)
        x = self.classifier(x)

        return x
