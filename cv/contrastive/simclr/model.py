import torch.nn as nn
from typing import Any, Union

from cv import backbones
from cv.utils import MetaWrapper

class SimCLR(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for SimCLR architecture from paper on: A Simple Framework for Contrastive Learning of Visual Representations"

    def __init__(
            self,
            backbone: Union[nn.Module, str],
            backbone_params: dict[str, Any],
            projection_dim: int
    ):
        
        super(SimCLR, self).__init__()

        if isinstance(backbone, str):
            encoder = getattr(backbones, backbone)(**backbone_params)

        self.encoder = encoder.backbone
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=projection_dim)
        )
        
    def forward(self, x):

        x = self.encoder(x)
        x = self.projection(x)

        return x
