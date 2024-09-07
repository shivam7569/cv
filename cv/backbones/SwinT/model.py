import torch.nn as nn
from itertools import islice

from cv.utils import Global
from cv.utils.layers import PatchMerge
from cv.utils import MetaWrapper
from cv.attention.transformers import SwinTransformer


class SwinT(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for SWin Transformer architecture from paper on: Hierarchical Vision Transformer using Shifted Windows"

    def __init__(
            self,
            embed_c=192,
            patch_size=4,
            window_size=7,
            d_ff_ratio=4,
            num_heads_per_stage=[6, 12, 24, 48],
            shift=2,
            patch_merge_size=2,
            stage_embed_dim_ratio=2,
            num_blocks=[2, 2, 18, 2],
            classifier_mlp_d=2048,
            encoder_dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            classifier_dropout=0.0,
            global_aggregate="avg",
            image_size=224,
            patchify_technique="linear",
            stodepth=False,
            stodepth_mp=0.0,
            layer_scale=None,
            qkv_bias=False,
            in_channels=3,
            num_classes=1000
        ):

        super(SwinT, self).__init__()

        SwinT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels
        self.window_size = window_size
        self.num_blocks = num_blocks

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=embed_c),
            nn.LayerNorm(normalized_shape=embed_c)
        )

        if stodepth:
            drop_probs = [stodepth_mp / (sum(num_blocks) - 1) * i for i in range(sum(num_blocks))]
        else:
            drop_probs = [0.0 for _ in range(sum(num_blocks))]
        
        drop_probs = [list(islice(drop_probs, sum(num_blocks[:i]), sum(num_blocks[:i+1]))) for i in range(len(num_blocks))]

        common_parameters = {
            "d_ff_ratio": d_ff_ratio,
            "window_size": window_size,
            "shift": shift,
            "encoder_dropout": encoder_dropout,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "layer_scale": layer_scale,
            "qkv_bias": qkv_bias
        }

        swin_transformer = []
        for i in range(len(num_blocks)):
            stage = SwinTBlock(
                embed_c=embed_c*(stage_embed_dim_ratio**i), num_blocks=num_blocks[i],
                stodepth_probs=drop_probs[i], num_heads=num_heads_per_stage[i], **common_parameters
            )
            patch_merger = PatchMerge(m=patch_merge_size, in_c=embed_c*(stage_embed_dim_ratio**i)) if i != len(num_blocks) - 1 else nn.Identity()
            swin_transformer.extend([stage, patch_merger])

        self.swin_transformer = nn.Sequential(*swin_transformer)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)) if global_aggregate == "avg" else nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=embed_c*(stage_embed_dim_ratio**(len(num_blocks) - 1)), out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

    @staticmethod
    def _assertions(image_size, patch_size, patchify_technique):
        assert image_size % patch_size == 0, Global.LOGGER.error(f"Patch size {patch_size} is not a divisor of image dimension {image_size}")
        assert patchify_technique in ("linear", "convolutional"), Global.LOGGER.error(f"{patchify_technique} patchify technique not supported")

    def _linear_patchify(self, img):

        B, _, H, W = img.shape

        linear_projection = nn.functional.unfold(
            input=img, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )
        linear_projection = linear_projection.transpose(dim0=1, dim1=2)
        linear_projection = linear_projection.view(B, H // self.patch_size, W // self.patch_size, -1)

        return linear_projection

    def _conv_patchify(self, img):
        self.conv_projection_layer = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size
        ).to(img.device)

        conv_projection = self.conv_projection_layer(img).permute(0, 2, 3, 1)

        return conv_projection

    def forward(self, x):

        x = self.patchify(x)
        x = self.linear_projection(x)
        x = self.swin_transformer(x)
        x = self.classifier(x.permute(0, 3, 1, 2))

        return x

class SwinTBlock(nn.Module):

    def __init__(
            self,
            embed_c,
            d_ff_ratio,
            num_heads,
            window_size,
            shift,
            num_blocks,
            encoder_dropout,
            attention_dropout,
            projection_dropout,
            stodepth_probs,
            layer_scale,
            qkv_bias
    ):
        
        super(SwinTBlock, self).__init__()

        block_transformers = [
            SwinTransformer(
                embed_c=embed_c,
                d_ff_ratio=d_ff_ratio,
                num_heads=num_heads,
                window_size=window_size,
                shift=shift if i % 2 != 0 else 0,
                encoder_dropout=encoder_dropout,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
                stodepth_prob=stodepth_probs[i],
                layer_scale=layer_scale,
                qkv_bias=qkv_bias
            )
            for i in range(num_blocks)
        ]

        self.block_transformers = nn.Sequential(*block_transformers)

    def forward(self, x):
        return self.block_transformers(x)
