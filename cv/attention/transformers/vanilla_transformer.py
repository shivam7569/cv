import torch.nn as nn

from cv.attention.feed_forwards import LeFF
from cv.utils import MetaWrapper
from cv.attention.variants import MultiHeadSelfAttention
from cv.utils.layers import DropPath, LayerScale, TransformerSEBlock

class TransformerBlock(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Transformer Block with the ability to adapt from paper on: An image is worth 16x16 words"

    def __init__(
            self, embed_dim, d_ff, num_heads, encoder_dropout, attention_dropout, projection_dropout,
            ln_order="post", stodepth_prob=0.0, layer_scale=None, se_block=None, se_points="both",
            qkv_bias=False, in_dims=None, re_attention=False, leff=None, ceit_cross_attention=False
        ):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ln_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            attention_dropout=attention_dropout, projection_dropout=projection_dropout,
            qkv_bias=qkv_bias, in_dims=in_dims, re_attention=re_attention
        )

        if leff is None:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=embed_dim, out_features=d_ff),
                nn.GELU(),
                nn.Dropout(p=encoder_dropout),
                nn.Linear(in_features=d_ff, out_features=embed_dim)
            )
        else:
            self.mlp = LeFF(**leff)

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

        self.ceit_cross_attention = ceit_cross_attention

    def forward_post_ln(self, x, mask=None):

        if not self.ceit_cross_attention:
            msa_out = self.msa(q=x, k=x, v=x, mask=mask)
        else:
            x, other_layers_class_tokens = x[:, -1, :].unsqueeze(1), x[:, :-1, :]
            msa_out = self.msa(q=x, k=other_layers_class_tokens, v=other_layers_class_tokens, mask=mask)

        if self.se_points in ["both", "msa"]:
            msa_out = self.se_block(msa_out)

        msa_out = self.dropPath_1(self.ls1(msa_out))

        msa_residual = x + msa_out

        ln_1_out = self.ln_1(msa_residual)

        mlp_out = self.mlp(ln_1_out)

        if self.se_points in ["both", "mlp"]:
            mlp_out = self.se_block(mlp_out)

        mlp_out = self.dropPath_2(self.ls2(mlp_out))

        mlp_residual = ln_1_out + mlp_out

        ln_2_out = self.ln_2(mlp_residual)

        return ln_2_out

    def forward_pre_ln(self, x, mask=None):

        ln_1_out = self.ln_1(x)

        if not self.ceit_cross_attention:
            msa_out = self.msa(q=ln_1_out, k=ln_1_out, v=ln_1_out, mask=mask)
        else:
            x, other_layers_class_tokens = self.ln_1(x[:, -1, :].unsqueeze(1)), self.ln_1(x[:, :-1, :])
            msa_out = self.msa(q=x, k=other_layers_class_tokens, v=other_layers_class_tokens, mask=mask)
        
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

    def forward_dual_residual(self, x, res, mask=None):

        if not self.ceit_cross_attention:
            msa_out = self.msa(q=x, k=x, v=x, mask=mask)
        else:
            x, other_layers_class_tokens = x[:, -1, :].unsqueeze(1), x[:, :-1, :]
            msa_out = self.msa(q=x, k=other_layers_class_tokens, v=other_layers_class_tokens, mask=mask)

        if self.se_points in ["both", "msa"]:
            msa_out = self.se_block(msa_out)
        
        msa_out = self.dropPath_1(self.ls1(msa_out))

        x = x + msa_out
        x = self.ln_1(x)

        res = msa_out + res

        mlp_out = self.mlp(x)

        if self.se_points in ["both", "mlp"]:
            mlp_out = self.se_block(mlp_out)
        
        mlp_out = self.dropPath_2(self.ls2(mlp_out))

        x = x + mlp_out
        x = self.ln_2(x)

        res = mlp_out + res

        return (x, res)

    def forward(self, x, res=None, mask=None):
        if self.ln_order == "post":
            return self.forward_post_ln(x, mask)
        elif self.ln_order == "pre":
            return self.forward_pre_ln(x, mask)
        elif self.ln_order == "residual":
            assert res is not None
            return self.forward_dual_residual(x, res, mask)
        