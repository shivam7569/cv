import math
import torch
import numpy as np
import torch.nn as nn

from utils.global_params import Global

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, qkv_bias=False, in_dims=None):
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

        self.attention_dropout = nn.Dropout(p=attention_dropout)

    def _init_qkv_weights(self):

        nn.init.trunc_normal_(self.w_q.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_v.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_o.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_o.bias)

    def attention(self, query, key, value, mask):

        enery = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            enery.masked_fill_(mask == 0, np.NINF)

        attention_scores = enery.softmax(dim=-1)

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

        msa_out = self.w_o(attention_out) # shape: (b, sl, ed)

        return msa_out
