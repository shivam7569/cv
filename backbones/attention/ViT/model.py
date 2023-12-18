import torch
import torch.nn as nn
from backbones.attention import ViTEncoder

from utils.global_params import Global

class ViT(nn.Module):

    def __init__(
            self,
            num_classes,
            d_model,
            image_size,
            patch_size,
            classifier_mlp_d,
            encoder_mlp_d,
            encoder_num_heads,
            num_encoder_blocks,
            dropout=0.1,
            encoder_dropout=None,
            encoder_attention_dropout=None,
            patchify_technique="linear",
            in_channels=3
        ):
        
        super(ViT, self).__init__()

        ViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Linear(in_features=self.embed_size, out_features=d_model)

        self.class_token = nn.Parameter(
            torch.rand((1, 1, d_model), device="cuda:0"), requires_grad=True
        )

        self.position_embeddings = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, 1, d_model), device="cuda:0"), requires_grad=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, attention_dropout=encoder_attention_dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

        self.linear_projection.to("cuda:0")
        self.encoder.to("cuda:0")
        self.classifier.to("cuda:0")
        self.dropout.to("cuda:0")

    @staticmethod
    def _assertions(image_size, patch_size, patchify_technique):
        assert image_size % patch_size == 0, Global.LOGGER.error(f"Patch size {patch_size} is not a divisor of image dimension {image_size}")
        assert patchify_technique in ("linear", "convolutional"), Global.LOGGER.error(f"{patchify_technique} patchify technique not supported")

    def _linear_patchify(self, img):
        linear_projection = nn.functional.unfold(
            input=img, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )
        linear_projection = linear_projection.transpose(dim0=1, dim1=2)

        return linear_projection

    def _conv_patchify(self, img):
        conv_projection = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2)

        return conv_projection

    def forward(self, x):
        x = self.patchify(x)
        x = self.linear_projection(x)

        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.classifier(x)

        return x[:, 0, :]
    