import torch
import torch.nn as nn

from utils.layers import ConvLayerNorm, DropPath, LayerScale

class ConvNeXtParams:

    NUM_CLASSES: int = 1000
    IN_CHANNELS: int = 3
    STEM_OUT_CHANNELS: int = 96
    STEM_KERNEL_SIZE: int = 4
    STEM_KERNEL_STRIDE: int = 4
    NUM_BLOCKS: list[int] = [3, 3, 9, 3]
    EXPANSION_RATE: int = 4
    DEPTHWISE_CONV_KERNEL_SIZE: int = 7
    LAYER_SCALE: float = 1e-6
    STOCHASTIC_DEPTH_MP: float = 0.1

    @classmethod
    def setParams(cls, **kwargs):
        cls.NUM_CLASSES = kwargs["num_classes"]
        cls.IN_CHANNELS = kwargs["in_channels"]
        cls.STEM_OUT_CHANNELS = kwargs["stem_out_channels"]
        cls.STEM_KERNEL_SIZE = kwargs["stem_kernel_size"]
        cls.STEM_KERNEL_STRIDE = kwargs["stem_kernel_stride"]
        cls.NUM_BLOCKS = kwargs["num_blocks"]
        cls.EXPANSION_RATE = kwargs["expansion_rate"]
        cls.DEPTHWISE_CONV_KERNEL_SIZE = kwargs["depthwise_conv_kernel_size"]
        cls.LAYER_SCALE = kwargs["layer_scale"]
        cls.STOCHASTIC_DEPTH_MP = kwargs["stochastic_depth_mp"]

class ConvNeXt(nn.Module):

    def __init__(self, **kwargs):
        super(ConvNeXt, self).__init__()

        if len(kwargs): ConvNeXtParams.setParams(**kwargs)

        if ConvNeXtParams.STOCHASTIC_DEPTH_MP is not None:
            drop_probs = [ConvNeXtParams.STOCHASTIC_DEPTH_MP / (sum(ConvNeXtParams.NUM_BLOCKS) - 1) * i for i in range(sum(ConvNeXtParams.NUM_BLOCKS))]
        else:
            drop_probs = [0.0 for _ in range(sum(ConvNeXtParams.NUM_BLOCKS))]

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=ConvNeXtParams.IN_CHANNELS, out_channels=ConvNeXtParams.STEM_OUT_CHANNELS,
                kernel_size=ConvNeXtParams.STEM_KERNEL_SIZE, stride=ConvNeXtParams.STEM_KERNEL_STRIDE,
                padding=0, groups=1
            ),
            ConvLayerNorm(normalized_shape=ConvNeXtParams.STEM_OUT_CHANNELS, data_format="channels_first")
        )

        self.convnext_group_1 = ConvNeXtGroup(
            num_blocks=ConvNeXtParams.NUM_BLOCKS[0], in_channels=ConvNeXtParams.STEM_OUT_CHANNELS,
            expansion_rate=ConvNeXtParams.EXPANSION_RATE, downsample=False, drop_probs=drop_probs[0: sum(ConvNeXtParams.NUM_BLOCKS[:1])]
        )

        self.convnext_group_2 = ConvNeXtGroup(
            num_blocks=ConvNeXtParams.NUM_BLOCKS[1], in_channels=ConvNeXtParams.STEM_OUT_CHANNELS * 2,
            expansion_rate=ConvNeXtParams.EXPANSION_RATE, downsample=True, drop_probs=drop_probs[sum(ConvNeXtParams.NUM_BLOCKS[:1]): sum(ConvNeXtParams.NUM_BLOCKS[:2])]
        )

        self.convnext_group_3 = ConvNeXtGroup(
            num_blocks=ConvNeXtParams.NUM_BLOCKS[2], in_channels=ConvNeXtParams.STEM_OUT_CHANNELS * 4,
            expansion_rate=ConvNeXtParams.EXPANSION_RATE, downsample=True, drop_probs=drop_probs[sum(ConvNeXtParams.NUM_BLOCKS[:2]): sum(ConvNeXtParams.NUM_BLOCKS[:3])]
        )

        self.convnext_group_4 = ConvNeXtGroup(
            num_blocks=ConvNeXtParams.NUM_BLOCKS[3], in_channels=ConvNeXtParams.STEM_OUT_CHANNELS * 8,
            expansion_rate=ConvNeXtParams.EXPANSION_RATE, downsample=True, drop_probs=drop_probs[sum(ConvNeXtParams.NUM_BLOCKS[:3]): sum(ConvNeXtParams.NUM_BLOCKS[:4])]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(normalized_shape=ConvNeXtParams.STEM_OUT_CHANNELS * 8),
            nn.Linear(in_features=ConvNeXtParams.STEM_OUT_CHANNELS * 8, out_features=ConvNeXtParams.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.convnext_group_1(x)
        x = self.convnext_group_2(x)
        x = self.convnext_group_3(x)
        x = self.convnext_group_4(x)
        x = self.classifier(x)

        return x

class ConvNeXtGroup(nn.ModuleList):
    def __init__(self, num_blocks, in_channels, expansion_rate, downsample, drop_probs):
        super(ConvNeXtGroup, self).__init__()

        if downsample:
            downsample_layer = ConvBlock(
                in_channels=in_channels // 2, out_channels=in_channels,
                kernel_size=2, stride=2, padding=0, groups=1
            )

            self.append(downsample_layer)

        for i in range(num_blocks):
            block = ConvNeXtBlock(
                in_channels=in_channels, expansion_rate=expansion_rate, drop_prob=drop_probs[i]
            )
            self.append(block)

    def forward(self, x):
        for layer in self:
            x = layer(x)

        return x
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, expansion_rate, drop_prob):
        
        super(ConvNeXtBlock, self).__init__()

        self.depth_wise_conv = ConvBlock(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=ConvNeXtParams.DEPTHWISE_CONV_KERNEL_SIZE, stride=1, padding=3, groups=in_channels
        )
        self.point_wise_expansion = ConvBlock(
            in_channels=in_channels, out_channels=expansion_rate*in_channels,
            kernel_size=1, stride=1, padding=0, groups=1
        )
        self.point_wise_conv = ConvBlock(
            in_channels=expansion_rate*in_channels, out_channels=in_channels,
            kernel_size=1, stride=1, padding=0, groups=1
        )

        self.ln = ConvLayerNorm(normalized_shape=in_channels, data_format="channels_first")
        self.gelu = nn.GELU()

        self.layer_scale = LayerScale(
            num_channels=in_channels,
            init_value=ConvNeXtParams.LAYER_SCALE
        ) if ConvNeXtParams.LAYER_SCALE is not None else None

        self.dropPath = DropPath(drop_prob=drop_prob)

    def forward(self, x):
        block_out = self.depth_wise_conv(x)
        block_out = self.ln(block_out)
        block_out = self.point_wise_expansion(block_out)
        block_out = self.gelu(block_out)
        block_out = self.point_wise_conv(block_out)

        if self.layer_scale is not None:
            block_out = self.layer_scale(block_out)

        block_out = self.dropPath(block_out)

        res = torch.add(x, block_out)

        return res
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
        )

        self.initializeConv()

    def initializeConv(self):
        nn.init.trunc_normal_(self.conv.weight, mean=0, std=0.2)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)

        return x
