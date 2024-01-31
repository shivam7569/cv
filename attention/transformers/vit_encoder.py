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
            in_dims=None
        ):

        super(ViTEncoder, self).__init__()

        if stodepth:
            drop_probs = [stodepth_mp / (num_blocks - 1) * i for i in range(num_blocks)]
        else:
            drop_probs = [0.0 for _ in range(num_blocks)]

        encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads,
                    encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                    ln_order=ln_order, stodepth_prob=drop_probs[i], layer_scale=layer_scale,
                    se_block=se_block, se_points=se_points, qkv_bias=qkv_bias, in_dims=in_dims
                ) for i in range(num_blocks)
            ]
        )
        
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):

        encoder_out = self.encoder(x)

        return encoder_out
