import math
import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper

class WindowAttention(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Window Attention Block from paper on: Hierarchical Vision Transformer using Shifted Windows"

    def __init__(
            self,
            embed_dim,
            num_heads,
            window_size,
            shift,
            qkv_bias=False,
            attention_dropout=0.0,
            projection_dropout=0.0
        ):

        super(WindowAttention, self).__init__()

        assert embed_dim % num_heads == 0, Global.LOGGER.error(f"Number of attention heads {num_heads} is not a divisor of embedding dimension: {embed_dim}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift

        self.d_k = embed_dim // num_heads

        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.attention_dropout = nn.Dropout(p=attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.projection_dropout = nn.Dropout(p=projection_dropout) if projection_dropout > 0.0 else nn.Identity()

        self.relative_position_table = nn.Parameter(
            torch.zeros(
                size=((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
        )
        nn.init.trunc_normal_(self.relative_position_table, mean=0.0, std=0.02)

        self.register_buffer(
            "relative_position_index", self._get_relative_position_index(), persistent=False
        )

        self._init_qkv_weights()

    def _init_qkv_weights(self):

        nn.init.trunc_normal_(self.w_q.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_k.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_v.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_o.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w_o.bias)

    def _get_relative_position_index(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)

        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, start_dim=1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1

        relative_position_index = relative_coords.sum(dim=-1)

        return relative_position_index
    
    def _get_relative_position_bias(self):
        relative_position_bias = self.relative_position_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size ** 2, self.window_size ** 2, -1
        ).permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias
    
    def attention(self, query, key, value, mask=None):

        _, _, L, _ = query.shape

        energy = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        energy = energy + self._get_relative_position_bias()

        if mask is not None:
            num_windows = mask.shape[0]
            energy = energy.view(-1, num_windows, self.num_heads, L, L) + mask.unsqueeze(1).unsqueeze(0)
            energy = energy.view(-1, self.num_heads, L, L)

        attention_scores = energy.softmax(dim=-1)

        if self.attention_dropout is not None:
            attention_scores = self.attention_dropout(attention_scores)

        attention_out = attention_scores @ value

        return attention_out, attention_scores
    
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        attention_out, _ = self.attention(query=query, key=key, value=value, mask=mask)

        attention_out = attention_out.transpose(1, 2).contiguous().view(attention_out.shape[0], -1, self.embed_dim)

        wmsa_out = self.projection_dropout(self.w_o(attention_out))

        return wmsa_out
