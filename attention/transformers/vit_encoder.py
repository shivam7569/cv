import torch
import torch.nn as nn

from attention.transformers.vanilla_transformer import TransformerBlock

class ViTEncoder(nn.Module):

    def __init__(
            self,
            embed_dim,
            d_ff,
            num_heads,
            num_blocks,
            encoder_dropout=0.0,
            attention_dropout=0.0,
            ln_order="pre",
            stodepth=False,
            stodepth_mp=None,
            layer_scale=None,
            se_block=None,
            se_points="both",
            qkv_bias=False,
            in_dims=None,
            hvt_pool=None
        ):

        super(ViTEncoder, self).__init__()

        if stodepth:
            drop_probs = [stodepth_mp / (num_blocks - 1) * i for i in range(num_blocks)]
        else:
            drop_probs = [0.0 for _ in range(num_blocks)]

        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads,
                    encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                    ln_order=ln_order, stodepth_prob=drop_probs[i], layer_scale=layer_scale,
                    se_block=se_block, se_points=se_points, qkv_bias=qkv_bias, in_dims=in_dims
                ) for i in range(num_blocks)
            ]
        )
        
        self.encoder = nn.Sequential(*self.encoder_layers)

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
        else:
            if self.ln_order in ["post", "pre"]:
                encoder_out = self.encoder(x)

            elif self.ln_order == "residual":
                res = x.clone()
                for transformer_block in self.encoder_layers:
                    x, res = transformer_block(x, res)

                encoder_out = x + self.final_norm(res)

        return encoder_out
