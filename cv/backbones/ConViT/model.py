import torch
import torch.nn as nn
from cv.attention.transformers.vit_encoder import ViTEncoder

from cv.utils import Global
from cv.utils.layers import DropPath
from cv.utils import MetaWrapper

class ConViT(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for ConViT architecture from paper on: Improving Vision Transformers with Soft Convolutional Inductive Biases"

    def __init__(
            self,
            d_model,
            image_size,
            patch_size,
            classifier_mlp_d,
            encoder_mlp_d,
            encoder_num_heads,
            num_encoder_blocks,
            num_gated_blocks=10,
            locality_strength=1.0,
            locality_distance_method="constant",
            use_conv_init=True,
            d_pos=3,
            dropout=0.0,
            encoder_dropout=0.0,
            encoder_attention_dropout=0.0,
            encoder_projection_dropout=0.0,
            patchify_technique="linear",
            stochastic_depth=False,
            stochastic_depth_mp=None,
            layer_scale=None,
            ln_order="residual",
            in_channels=3,
            num_classes=1000
        ):
        
        super(ConViT, self).__init__()

        ConViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=d_model),
            nn.LayerNorm(normalized_shape=d_model)
        )

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout,
            stodepth=stochastic_depth, stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order,
            hvt_pool=None, gated_transformer_params={
                "locality_strength": locality_strength,
                "locality_distance_method": locality_distance_method,
                "use_conv_init": use_conv_init,
                "d_pos": d_pos,
                "num_blocks": num_gated_blocks
            }
        )
        self.stochastic_depth_mp = stochastic_depth_mp

        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

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

    def updateStochasticDepthRate(self, k=0.05):
        for module_name, encoder_module in self.encoder.named_modules():
            if isinstance(encoder_module, DropPath):
                encoder_number = int(module_name.split(".")[1])
                encoder_module.drop_prob = encoder_module.drop_prob + encoder_number * (k / (len(self.encoder.encoder) - 1))

    def forward(self, x):
        x = self.patchify(x)
        x = self.linear_projection(x)
        x = self.dropout(x)
        x = self.encoder(x)

        class_token = x[:, 0, :]

        x = self.classifier(class_token)

        return x
