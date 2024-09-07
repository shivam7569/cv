import torch
import torch.nn as nn

from cv.utils import MetaWrapper
from cv.attention.transformers import GatedTransformer, TransformerBlock

class ViTEncoder(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "ViT Encoder to culminate transformer blocks of different variations"

    def __init__(
            self,
            embed_dim,
            d_ff,
            num_heads,
            num_blocks,
            encoder_dropout=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            ln_order="pre",
            stodepth=False,
            stodepth_mp=None,
            layer_scale=None,
            se_block=None,
            se_points="both",
            qkv_bias=False,
            in_dims=None,
            hvt_pool=None,
            gated_transformer_params=None,
            re_attention=False,
            leff_params=None
        ):

        super(ViTEncoder, self).__init__()

        if stodepth:
            drop_probs = [stodepth_mp / (num_blocks - 1) * i for i in range(num_blocks)]
        else:
            drop_probs = [0.0 for _ in range(num_blocks)]

        self.gated_transformer_params = gated_transformer_params
        self.leff_params = leff_params

        if gated_transformer_params is None:
            self.encoder_layers = nn.ModuleList(
                [
                    TransformerBlock(
                        embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads,
                        encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                        projection_dropout=projection_dropout, ln_order=ln_order, stodepth_prob=drop_probs[i], layer_scale=layer_scale,
                        se_block=se_block, se_points=se_points, qkv_bias=qkv_bias, in_dims=in_dims, re_attention=re_attention, leff=leff_params
                    ) for i in range(num_blocks)
                ]
            )
            
            self.encoder = nn.Sequential(*self.encoder_layers)
        else:
            self.class_token = nn.Parameter(
                torch.rand((1, 1, embed_dim)), requires_grad=True
            )
            nn.init.trunc_normal_(tensor=self.class_token, mean=0.0, std=0.02)
            self.encoder_layers = nn.ModuleList()
            for i in range(num_blocks):
                if i < gated_transformer_params["num_blocks"]:
                    self.encoder_layers.append(
                        GatedTransformer(
                            embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads,
                            encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout, locality_strength=gated_transformer_params["locality_strength"],
                            use_conv_init=gated_transformer_params["use_conv_init"], locality_distance_method=gated_transformer_params["locality_distance_method"],
                            d_pos=gated_transformer_params["d_pos"], ln_order=ln_order, stodepth_prob=drop_probs[i], layer_scale=layer_scale, se_block=se_block,
                            se_points=se_points, qkv_bias=qkv_bias, in_dims=in_dims
                        )
                    )
                else:
                    self.encoder_layers.append(
                        TransformerBlock(
                            embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads,
                            encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                            projection_dropout=projection_dropout, ln_order=ln_order, stodepth_prob=drop_probs[i],
                            layer_scale=layer_scale, se_block=se_block, se_points=se_points, qkv_bias=qkv_bias, in_dims=in_dims
                        )
                    )

        self.ln_order = ln_order

        if self.ln_order == "residual":
            self.final_norm = nn.LayerNorm(normalized_shape=embed_dim)

        self.hvt_pool = hvt_pool
        if self.hvt_pool is not None:
            self.max_pool_layer = nn.MaxPool1d(kernel_size=3, stride=2)
            self.position_embeddings = nn.Parameter(
                torch.zeros(size=(1, 1, embed_dim)), requires_grad=True
            )
            nn.init.trunc_normal_(tensor=self.position_embeddings, mean=0.0, std=0.02)
            self.average_pool_layer = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):

        if self.hvt_pool is not None:
            if self.ln_order in ["post", "pre"]:
                for idx, transformer_block in enumerate(self.encoder_layers):
                    if idx in self.hvt_pool:
                        x = self.max_pool_layer(x.transpose(-1, -2)).transpose(-1, -2)
                        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
                    x = transformer_block(x)
                encoder_out = self.average_pool_layer(x.transpose(-1, -2)).transpose(-1, -2)
            elif self.ln_order == "residual":
                res = x.clone()
                for idx, transformer_block in enumerate(self.encoder_layers):
                    if idx in self.hvt_pool:
                        x = self.max_pool_layer(x.transpose(-1, -2)).transpose(-1, -2)
                        res = self.max_pool_layer(res.transpose(-1, -2)).transpose(-1, -2)
                        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
                        res = res + self.position_embeddings.expand(res.size(0), res.size(1), -1)
                    x, res = transformer_block(x, res)
                encoder_out = self.average_pool_layer(x.transpose(-1, -2)).transpose(-1, -2)
        elif self.gated_transformer_params is not None:
            if self.ln_order in ["post", "pre"]:
                for idx, transformer_block in enumerate(self.encoder_layers):
                    if idx == self.gated_transformer_params["num_blocks"]:
                        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
                    x = transformer_block(x)
                encoder_out = x
            elif self.ln_order == "residual":
                res = x.clone()
                for transformer_block in self.encoder_layers:
                    if (idx + 1) == self.gated_transformer_params["num_blocks"]:
                        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
                        res = torch.cat([self.class_token.expand(res.size(0), -1, -1), res], dim=1)
                    x, res = transformer_block(x, res)
                encoder_out = x + self.final_norm(res)
        elif self.leff_params is not None:
            lca = []
            for transformer_block in self.encoder:
                x = transformer_block(x)
                lca.append(x[:, 0, :])

            encoder_out = lca
        else:
            if self.ln_order in ["post", "pre"]:
                encoder_out = self.encoder(x)

            elif self.ln_order == "residual":
                res = x.clone()
                for transformer_block in self.encoder_layers:
                    x, res = transformer_block(x, res)

                encoder_out = x + self.final_norm(res)

        return encoder_out
