import math
import torch
import torch.nn as nn

from cv.utils import MetaWrapper

class LeFF(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Locally Enhanced Feed Forward Block from paper on: Incorporating Convolution Designs into Visual Transformers"

    def __init__(
            self,
            num_tokens,
            in_features,
            expand_ratio,
            depthwise_kernel,
            depthwise_stride,
            depthwise_padding,
            depthwise_separable
    ):

        super(LeFF, self).__init__()

        self.num_tokens = num_tokens
        self.in_features = in_features
        self.expand_ratio = expand_ratio

        self.expansion = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=expand_ratio * in_features
            ),
            nn.BatchNorm1d(num_features=num_tokens - 1),
            nn.GELU()
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=expand_ratio * in_features,
                out_channels=expand_ratio * in_features,
                kernel_size=depthwise_kernel,
                stride=depthwise_stride,
                padding=depthwise_padding,
                groups=expand_ratio * in_features
            ),
            nn.Conv2d(
                in_channels=expand_ratio * in_features,
                out_channels=expand_ratio * in_features,
                kernel_size=1,
                stride=1
            ) if depthwise_separable else nn.Identity(),
            nn.BatchNorm2d(num_features=expand_ratio * in_features),
            nn.GELU()
        )

        self.reduction = nn.Sequential(
            nn.Linear(
                in_features=expand_ratio * in_features,
                out_features=in_features
            ),
            nn.BatchNorm1d(num_features=num_tokens - 1),
            nn.GELU()
        )

    def forward(self, x):
        class_token, patch_tokens = x[:, 0, :], x[:, 1:, :]

        x = self.expansion(patch_tokens)
        x = x.transpose(1, 2).contiguous().reshape(-1, self.expand_ratio * self.in_features, int(math.sqrt(self.num_tokens)), int(math.sqrt(self.num_tokens)))
        x = self.depthwise_conv(x)
        x = x.reshape(-1, self.expand_ratio * self.in_features, int(math.sqrt(self.num_tokens)) ** 2).transpose(1, 2)
        x = self.reduction(x)

        x = torch.cat([class_token.unsqueeze(1), x], dim=1)

        return x
    