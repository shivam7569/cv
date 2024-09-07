import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv.utils import MetaWrapper
from cv.attention.transformers import ViTEncoder, TokenTransformer
from cv.utils.position_embeddings import get_sinusoidal_embedding

class T2T_ViT(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for Tokens to Tokens Transformer architecture from paper on: Training Vision Transformers from Scratch on ImageNet"

    def __init__(
            self,
            embed_dim=384,
            t2t_module_embed_dim=64,
            t2t_module_d_ff=64,
            t2t_module_transformer_num_heads=1,
            t2t_module_transformer_encoder_dropout=0.0,
            t2t_module_transformer_attention_dropout=0.0,
            t2t_module_transformer_projection_dropout=0.0,
            t2t_module_patch_size=3,
            t2t_module_overlapping=1,
            t2t_module_padding=1,
            soft_split_kernel_size=7,
            soft_split_stride=4,
            soft_split_padding=2,
            vit_backbone_d_ff=1152,
            vit_backbone_num_heads=12,
            vit_backbone_num_blocks=14,
            vit_backbone_encoder_dropout=0.0,
            vit_backbone_attention_dropout=0.0,
            vit_backbone_projection_dropout=0.0,
            vit_backbone_stodepth=True,
            vit_backbone_stodepth_mp=0.1,
            vit_backbone_layer_scale=1e-6,
            vit_backbone_num_patches=196,
            vit_backbone_se_points="both",
            vit_backbone_qkv_bias=False,
            vit_backbone_in_dims=None,
            classifier_hidden_dim=2048,
            classifier_dropout=0.0,
            classifier_num_classes=1000,
            in_channels=3,
            image_size=224
            ):

        super(T2T_ViT, self).__init__()

        self.t2t_module = T2T_Module(
            embed_dim=t2t_module_embed_dim,
            projection_dim=embed_dim,
            t2t_module_d_ff=t2t_module_d_ff,
            t2t_module_transformer_num_heads=t2t_module_transformer_num_heads,
            t2t_module_transformer_encoder_dropout=t2t_module_transformer_encoder_dropout,
            t2t_module_transformer_attention_dropout=t2t_module_transformer_attention_dropout,
            t2t_module_transformer_projection_dropout=t2t_module_transformer_projection_dropout,
            t2t_module_patch_size=t2t_module_patch_size,
            t2t_module_overlapping=t2t_module_overlapping,
            t2t_module_padding=t2t_module_padding,
            soft_split_kernel_size=soft_split_kernel_size,
            soft_split_stride=soft_split_stride,
            soft_split_padding=soft_split_padding,
            in_channels=in_channels

        )

        self.class_token = nn.Parameter(
            torch.rand((1, 1, embed_dim)), requires_grad=True
        )

        self.position_embeddings = get_sinusoidal_embedding(vit_backbone_num_patches+1, embed_dim)

        self.register_buffer("position_encodings", self.position_embeddings)

        self.vit = ViTEncoder(
            embed_dim=embed_dim, d_ff=vit_backbone_d_ff, num_heads=vit_backbone_num_heads,
            num_blocks=vit_backbone_num_blocks, encoder_dropout=vit_backbone_encoder_dropout, 
            attention_dropout=vit_backbone_attention_dropout, projection_dropout=vit_backbone_projection_dropout,
            stodepth=vit_backbone_stodepth, stodepth_mp=vit_backbone_stodepth_mp, layer_scale=vit_backbone_layer_scale,
            se_block=vit_backbone_num_patches+1, se_points=vit_backbone_se_points, qkv_bias=vit_backbone_qkv_bias, in_dims=vit_backbone_in_dims
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=classifier_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features=classifier_hidden_dim, out_features=classifier_num_classes)
        )

        self.image_size = image_size

    def forward(self, x):
        x = self.t2t_module(x)
        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1).to(x.device)

        x = self.vit(x)

        class_token = x[:, 0, :]

        x = self.classifier(class_token)

        return x

class T2T_Module(nn.Module):

    def __init__(
            self,
            embed_dim,
            projection_dim,
            t2t_module_d_ff,
            t2t_module_transformer_num_heads,
            t2t_module_transformer_encoder_dropout,
            t2t_module_transformer_attention_dropout,
            t2t_module_transformer_projection_dropout,
            t2t_module_patch_size,
            t2t_module_overlapping,
            t2t_module_padding,
            soft_split_kernel_size,
            soft_split_stride,
            soft_split_padding,
            in_channels
        ):

        super(T2T_Module, self).__init__()

        self.image_soft_split = nn.Unfold(
            kernel_size=soft_split_kernel_size, stride=soft_split_stride, padding=soft_split_padding
        )

        self.t2t_transformer_1 = TokenTransformer(
            in_dims=(soft_split_kernel_size ** 2) * in_channels,
            embed_dim=embed_dim,
            d_ff=t2t_module_d_ff,
            num_heads=t2t_module_transformer_num_heads,
            encoder_dropout=t2t_module_transformer_encoder_dropout,
            attention_dropout=t2t_module_transformer_attention_dropout,
            projection_dropout=t2t_module_transformer_projection_dropout
        )

        self.t2t_transformer_2 = TokenTransformer(
            in_dims=(in_channels ** 2) * t2t_module_d_ff,
            embed_dim=embed_dim,
            d_ff=t2t_module_d_ff,
            num_heads=t2t_module_transformer_num_heads,
            encoder_dropout=t2t_module_transformer_encoder_dropout,
            attention_dropout=t2t_module_transformer_attention_dropout,
            projection_dropout=t2t_module_transformer_projection_dropout
        )

        self.projection_layer = nn.Linear(
            in_features=(in_channels ** 2) * t2t_module_d_ff, out_features=projection_dim
        )

        self.t2t_module_patch_size = t2t_module_patch_size
        self.t2t_module_overlapping = t2t_module_overlapping
        self.t2t_module_padding = t2t_module_padding

    def _t2t(self, x, patch_size, overlapping, padding):

        with torch.no_grad():
            h = w = int(math.sqrt(x.shape[1]))
    
        x = x.view(x.size(0), h, w, -1).transpose(dim0=1, dim1=3)

        x = F.unfold(
            input=x, kernel_size=patch_size,
            stride=int(patch_size-overlapping), padding=padding
        ).transpose(dim0=1, dim1=2)

        return x

    def forward(self, x):
        x = self.image_soft_split(x).transpose(dim0=1, dim1=2) # shape: (batch_size, patch_features, num_patches) -> (batch_size, num_patches, patch_features)

        x = self.t2t_transformer_1(x)
        x = self._t2t(x, patch_size=self.t2t_module_patch_size, overlapping=self.t2t_module_overlapping, padding=self.t2t_module_padding)
        x = self.t2t_transformer_2(x)
        x = self._t2t(x, patch_size=self.t2t_module_patch_size, overlapping=self.t2t_module_overlapping, padding=self.t2t_module_padding) # shape: (batch_size, num_patches===[14*14], patch_features)

        x = self.projection_layer(x)

        return x
