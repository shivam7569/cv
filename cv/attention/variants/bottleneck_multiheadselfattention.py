import math
import numpy as np
import torch.nn as nn
from einops import rearrange

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.utils.position_embeddings import RelPosEmb2D

class BottleNeck_MHSA(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Bottleneck Multi Head Self Attention from paper on: Bottleneck Transformers for Visual Recognition"

    def __init__(
            self,
            embed_dims,
            num_heads,
            feature_map_size,
            attention_dropout=0.0,
            qkv_bias=False
    ):
        
        super(BottleNeck_MHSA, self).__init__()

        assert embed_dims % num_heads == 0, Global.LOGGER.error(f"Number of attention heads {num_heads} is not a divisor of embedding dimension: {embed_dims}")

        self.embed_dim = embed_dims
        self.num_heads = num_heads

        self.d_k = embed_dims // num_heads

        self.w_q = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=qkv_bias
        )
        self.w_k = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=qkv_bias
        )
        self.w_v = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=qkv_bias
        )

        self.w_o = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.h, self.w = feature_map_size
        self.relative_position_embeddings = RelPosEmb2D(feature_map_size=feature_map_size, dim_head=self.d_k)
        
        self.attention_dropout = nn.Dropout(p=attention_dropout) if attention_dropout > 0.0 else nn.Identity()

    def attention(self, query, key, value, mask):

        energy = query @ key.transpose(-2, -1)
        content_aware_pos_enc = self.relative_position_embeddings(query)

        energy = (energy + content_aware_pos_enc) / math.sqrt(self.d_k)

        if mask is not None:
            energy.masked_fill_(mask == 0, np.NINF)

        attention_scores = energy.softmax(dim=-1)

        if self.attention_dropout is not None:
            attention_scores = self.attention_dropout(attention_scores)

        attention_out = attention_scores @ value

        return attention_out, attention_scores

    def forward(self, x, mask=None):

        q, k, v = x, x, x

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = rearrange(query, "b (dk h) hi wi -> b h (hi wi) dk", h=self.num_heads, dk=self.d_k)
        key = rearrange(key, "b (dk h) hi wi -> b h (hi wi) dk", h=self.num_heads, dk=self.d_k)
        value = rearrange(value, "b (dk h) hi wi -> b h (hi wi) dk", h=self.num_heads, dk=self.d_k)

        attention_out, _ = self.attention(query=query, key=key, value=value, mask=mask)

        attention_out = rearrange(attention_out, "b h (hi wi) d -> b (h d) hi wi", hi=self.h, wi=self.w)

        msa_out = self.w_o(attention_out)

        return msa_out
