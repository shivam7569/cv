import torch
import torch.nn as nn

from backbones.attention import MultiHeadSelfAttention

class ViTEncoder(nn.Module):

    def __init__(self, embed_dim, d_ff, num_heads, num_blocks, encoder_dropout=None, attention_dropout=None):
        super(ViTEncoder, self).__init__()

        encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim, d_ff=d_ff, num_heads=num_heads, encoder_dropout=encoder_dropout, attention_dropout=attention_dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):

        encoder_out = self.encoder(x)

        return encoder_out


class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, d_ff, num_heads, encoder_dropout, attention_dropout):
        super(EncoderBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ln_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attention_dropout=attention_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=d_ff),
            nn.GELU(),
            nn.Dropout(p=encoder_dropout),
            nn.Linear(in_features=d_ff, out_features=embed_dim)
        )

    def forward(self, x):
        ln_1_out = self.ln_1(x)
        msa_out = self.msa(q=ln_1_out, k=ln_1_out, v=ln_1_out, mask=None)

        msa_residual = x + msa_out

        ln_2_out = self.ln_2(msa_residual)
        mlp_out = self.mlp(ln_2_out)

        mlp_residual = msa_residual + mlp_out

        return mlp_residual
