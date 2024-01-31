import torch.nn as nn
from attention.variants.multiheadselfattention import MultiHeadSelfAttention
from utils.pytorch_utils import DropPath, LayerScale, TransformerSEBlock

class TransformerBlock(nn.Module):

    def __init__(
            self, embed_dim, d_ff, num_heads, encoder_dropout, attention_dropout,
            ln_order="post", stodepth_prob=0.0, layer_scale=None, se_block=None, se_points="both",
            qkv_bias=False, in_dims=None
        ):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ln_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            attention_dropout=attention_dropout, qkv_bias=qkv_bias, in_dims=in_dims
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=d_ff),
            nn.GELU(),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(in_features=d_ff, out_features=embed_dim)
        )

        self.ln_order = ln_order
        self.dropPath_1 = DropPath(
            drop_prob=stodepth_prob
        ) if stodepth_prob > 0.0 else nn.Identity()

        self.dropPath_2 = DropPath(
            drop_prob=stodepth_prob
        ) if stodepth_prob > 0.0 else nn.Identity()

        self.ls1 = LayerScale(num_channels=embed_dim, init_value=layer_scale, type_="msa") if layer_scale is not None else nn.Identity()
        self.ls2 = LayerScale(num_channels=embed_dim, init_value=layer_scale, type_="msa") if layer_scale is not None else nn.Identity()

        self.se_block = TransformerSEBlock(in_channels=se_block, r=16) if se_block is not None else nn.Identity()
        self.se_points = se_points

    def forward_post_ln(self, x):

        msa_out = self.msa(q=x, k=x, v=x, mask=None)
        msa_residual = x + msa_out

        ln_1_out = self.ln_1(msa_residual)

        mlp_out = self.mlp(ln_1_out)
        mlp_residual = ln_1_out + mlp_out

        ln_2_out = self.ln_2(mlp_residual)

        return ln_2_out

    def forward_pre_ln(self, x):

        ln_1_out = self.ln_1(x)
        msa_out = self.msa(q=ln_1_out, k=ln_1_out, v=ln_1_out, mask=None)

        if self.se_points in ["both", "msa"]:
            msa_out = self.se_block(msa_out)
        
        msa_out = self.dropPath_1(self.ls1(msa_out))

        msa_residual = x + msa_out

        ln_2_out = self.ln_2(msa_residual)
        mlp_out = self.mlp(ln_2_out)

        if self.se_points in ["both", "mlp"]:
            mlp_out = self.se_block(mlp_out)
        
        mlp_out = self.dropPath_2(self.ls2(mlp_out))

        mlp_residual = msa_residual + mlp_out

        return mlp_residual

    def forward(self, x):
        if self.ln_order == "post":
            return self.forward_post_ln(x)
        elif self.ln_order == "pre":
            return self.forward_pre_ln(x)
        