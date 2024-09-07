import torch
import torch.nn as nn
import torch.nn.functional as F

from cv.utils import MetaWrapper

class DropPath(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class implementing Stochastic Depth across model per layer basis"
    
    def __init__(self, drop_prob, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor
    
class LayerScale(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to implement Layer Scaling as used in ViT training"

    def __init__(self, num_channels, init_value, type_="conv"):
        super(LayerScale, self).__init__()

        if type_ == "conv":
            self.scale = nn.Parameter(
                init_value * torch.ones((1, num_channels, 1, 1))
            )
        elif type_ == "msa":
            self.scale = nn.Parameter(
                init_value * torch.ones((1, 1, num_channels))
            )

    def forward(self, x):
        return x * self.scale

class ConvLayerNorm(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to implement layer normalization for convolutional layers"
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class TransformerSEBlock(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to implement Squeeze and Excitation method for transformers"

    def __init__(self, in_channels, r=16):
        super(TransformerSEBlock, self).__init__()

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels, out_features=in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // r, out_features=in_channels, bias=False),
            nn.Sigmoid()
        )

        self.in_channels = in_channels

    def forward(self, x):
        squeeze = self.squeeze(x)
        excitation = squeeze.view(-1, self.in_channels, 1).expand_as(x) * x

        return excitation

class PatchMerge(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to implement Patch Merging as used in SWin Transformer"

    def __init__(self, m, in_c):
        super(PatchMerge, self).__init__()

        self.m = m
        self.in_c = in_c

        self.norm = nn.LayerNorm(normalized_shape=m*in_c)
        self.reduction = nn.Linear(in_features=(m**2)*in_c, out_features=m*in_c)

    def forward(self, x):

        coords = [(i, j) for i in range(self.m) for j in range(self.m)]

        distinct_patches = []

        for coord in coords:
            coord_window = x[:, coord[0]::self.m, coord[1]::self.m, :]
            distinct_patches.append(coord_window)

        merged_patches = torch.cat(distinct_patches, dim=-1)
        merged_patches = self.norm(self.reduction(merged_patches))

        return merged_patches
