import math
import torch
import numpy as np
import torch.nn as nn

from cv.utils import MetaWrapper

class Gated_MultiHeadSelfAttention(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Gated Multi Head Self Attention Block from paper on: Improving Vision Transformers with Soft Convolutional Inductive Biases"

    def __init__(self, embed_dim, num_heads, d_pos, locality_strength, use_conv_init,
                 locality_distance_method="constant", attention_dropout=0.0,
                 projection_dropout=0.0, qkv_bias=False):

        super(Gated_MultiHeadSelfAttention, self).__init__()

        assert embed_dim & num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_h = embed_dim // num_heads

        self.d_pos = d_pos

        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.attention_dropout = nn.Dropout(p=attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.projection_dropout = nn.Dropout(p=projection_dropout) if projection_dropout > 0.0 else nn.Identity()

        self.v_pos = nn.Linear(in_features=d_pos, out_features=num_heads)

        self.gating = nn.Parameter(torch.ones(num_heads), requires_grad=True)

        self.apply(self._init_linear_layers)

        if use_conv_init:
            self._conv_local_init(alpha=locality_strength, method=locality_distance_method)

    def _init_linear_layers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _conv_local_init(self, alpha, method):
        self.w_v.weight.data.copy_(torch.eye(n=self.embed_dim))
        locality_distance = 1 if method == "constant" else max(1, 1/math.sqrt(alpha))

        kernel_size = int(math.sqrt(self.num_heads))
        center = kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.v_pos.weight.data[position, 2] = -1
                self.v_pos.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.v_pos.weight.data[position, 0] = 2 * (h2 - center) * locality_distance

        self.v_pos.weight.data *= alpha

    def _create_relative_position_encodings(self, num_patches):
        vec_size = int(math.sqrt(num_patches))
        rel_indices = torch.zeros(1, num_patches, num_patches, self.d_pos)
        ind = torch.arange(vec_size).view(1, -1) - torch.arange(vec_size).view(-1, 1)
        indx = ind.repeat(vec_size, vec_size)
        indy = ind.repeat_interleave(vec_size, dim=0).repeat_interleave(vec_size, dim=1)

        indd = indx ** 2 + indy ** 2

        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)

        self.relative_position_embeddings = rel_indices.to(self.v_pos.weight.device)

        self.relative_position_embeddings.requires_grad = False
        self.register_buffer(name="relative_positions", tensor=self.relative_position_embeddings)

    def calculate_attention(self, query, key, value, mask):
        energy = (query @ key.transpose(-2, -1)) / math.sqrt(self.dim_h)

        if mask is not None:
            energy.masked_fill_(mask == 0, np.NINF)

        content_attention_scores = energy.softmax(dim=-1)

        position_projection = self.v_pos(self.relative_position_embeddings).permute(0, 3, 1, 2)
        position_attention_scores = position_projection.softmax(dim=-1)
        
        gating = self.gating.view(1, self.num_heads, 1, 1)

        attention_score = (1 - torch.sigmoid(gating)) * content_attention_scores + torch.sigmoid(gating) * position_attention_scores
        attention = attention_score / torch.sum(attention_score, dim=-1, keepdim=True) @ value
        attention = self.attention_dropout(attention)
        
        return attention

    def forward(self, q, k, v, mask):
        batch_size, num_patches, _ = q.shape

        if not hasattr(self, "relative_position_embeddings"):
            self._create_relative_position_encodings(num_patches=num_patches)
            self.relative_position_embeddings = self.relative_position_embeddings.expand(batch_size, -1, -1, -1)

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(batch_size, num_patches, self.num_heads, self.dim_h).transpose(1, 2)
        key = key.view(batch_size, num_patches, self.num_heads, self.dim_h).transpose(1, 2)
        value = value.view(batch_size, num_patches, self.num_heads, self.dim_h).transpose(1, 2)

        attention = self.calculate_attention(query, key, value, mask=mask)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, num_patches, -1)
        gmsa_out = self.w_o(attention)
        gmsa_out = self.projection_dropout(gmsa_out)

        return gmsa_out
