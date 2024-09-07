import math
import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper

class MultiHeadSelfAttention(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Multi Head Self Attention Block from paper on: An image is worth 16x16 words"

    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, projection_dropout=0.0, qkv_bias=False, in_dims=None, re_attention=False):
        super(MultiHeadSelfAttention, self).__init__()

        if in_dims is None: in_dims = embed_dim

        assert embed_dim % num_heads == 0, Global.LOGGER.error(f"Number of attention heads {num_heads} is not a divisor of embedding dimension: {embed_dim}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.d_k = embed_dim // num_heads

        self.w_q = nn.Linear(in_features=in_dims, out_features=embed_dim, bias=qkv_bias)
        self.w_k = nn.Linear(in_features=in_dims, out_features=embed_dim, bias=qkv_bias)
        self.w_v = nn.Linear(in_features=in_dims, out_features=embed_dim, bias=qkv_bias)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True)

        self.attention_dropout = nn.Dropout(p=attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.projection_dropout = nn.Dropout(p=projection_dropout) if projection_dropout > 0.0 else nn.Identity()

        self.re_attention = re_attention
        if self.re_attention:
            self.re_attention_layer = nn.Conv2d(
                in_channels=num_heads, out_channels=num_heads, kernel_size=1, stride=1
            )
            self.re_attention_bn = nn.BatchNorm2d(num_features=num_heads)

        self._init_qkv_weights()

    def _init_qkv_weights(self):

        nn.init.trunc_normal_(self.w_q.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_v.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_o.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_o.bias)

    def attention(self, query, key, value, mask):

        energy = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            energy.masked_fill_(mask, -torch.inf)

        attention_scores = energy.softmax(dim=-1)

        if self.re_attention:
            attention_scores = self.re_attention_layer(attention_scores)
            attention_scores = self.re_attention_bn(attention_scores)

        if self.attention_dropout is not None:
            attention_scores = self.attention_dropout(attention_scores)

        attention_out = attention_scores @ value

        return attention_out, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # shape: (batch_size, sequence_length, embed_dimension) = (b, sl, ed)
        key = self.w_k(k) # shape: (batch_size, sequence_length, embed_dimension) = (b, sl, ed)
        value = self.w_v(v) # shape: (batch_size, sequence_length, embed_dimension) = (b, sl, ed)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (b, sl, ed) -> (b, sl, num_heads, head_embed_dim) -> (b, num_heads, sl, head_embed_dim)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (b, sl, ed) -> (b, sl, num_heads, head_embed_dim) -> (b, num_heads, sl, head_embed_dim)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (b, sl, ed) -> (b, sl, num_heads, head_embed_dim) -> (b, num_heads, sl, head_embed_dim)
        
        attention_out, _ = self.attention(query=query, key=key, value=value, mask=mask)

        attention_out = attention_out.transpose(1, 2).contiguous().view(attention_out.shape[0], -1, self.embed_dim) # (b, num_heads, sl, head_embed_dim) -> (b, sl, num_heads, head_embed_dim) -> (b, sl, ed)

        msa_out = self.projection_dropout(self.w_o(attention_out)) # shape: (b, sl, ed)

        return msa_out
