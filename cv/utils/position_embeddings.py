import torch
import torch.nn as nn
from einops import rearrange

from cv.utils import MetaWrapper

class RelPosEmb1D(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class implementing Relative Position Embedding across 1 dimension"

    def __init__(self, seq_len, embed_dim_head, heads=None, scaled=False):

        super(RelPosEmb1D, self).__init__()

        scale = embed_dim_head ** -0.5 if scaled else 1
        self.shared_heads = heads if heads is not None else True

        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(2 * seq_len - 1, embed_dim_head) * scale
            )
        else:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(heads, 2 * seq_len - 1, embed_dim_head) * scale
            )

    def relative_to_absolute(self, emb):
        b, h, l, _, device, dtype = *emb.shape, emb.device, emb.dtype
        dd = {'device': device, 'dtype': dtype}
        col_pad = torch.zeros((b, h, l, 1), **dd)
        x = torch.cat((emb, col_pad), dim=3)
        flat_x = rearrange(x, 'b h l c -> b h (l c)')
        flat_pad = torch.zeros((b, h, l - 1), **dd)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
        final_x = final_x[:, :, :l, (l - 1):]
        return final_x

    def rel_pos_emb_1d(self, q, rel_emb):
        if self.shared_heads:
            emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
        else:
            emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
        return self.relative_to_absolute(emb)

    def forward(self, q):
        return self.rel_pos_emb_1d(q, self.rel_pos_emb)

class RelPosEmb2D(nn.Module):

    @classmethod
    def __class_repr__(cls):
        return "Class implementing Relative Position Embedding across 2 dimensions"

    def __init__(self, feature_map_size, dim_head, heads=None):

        super(RelPosEmb2D, self).__init__()

        self.h, self.w = feature_map_size
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads=heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads=heads)

    def expand_emb(self, r, dim_size):
        r = rearrange(r, "b (h x) i j -> b h x () i j", x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1, -1]
        r = r.expand(expand_index)

        return rearrange(r, "b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)")
    
    def forward(self, q):
        assert self.total_tokens == q.shape[2], f"Tokens {q.shape[2]} of q must be equal to the product \
            of the feature map size {self.total_tokens}"
        
        r_h = self.emb_w(rearrange(q, "b h (x y) d -> b (h x) y d", x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, "b h (x y) d -> b (h y) x d", x=self.h, y=self.w))

        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.w)

        return q_r
    
def get_sinusoidal_embedding(max_seq_len, embedding_dim):

    if embedding_dim % 2 != 0:
        raise ValueError(f"Sinusoidal position embeddings cannot be applied to odd token embedding dimension")

    position = torch.arange(0, max_seq_len).unsqueeze_(1)
    denominators = torch.pow(10000.0, 2*torch.arange(0, embedding_dim // 2) / 2)

    sinusoidal_embedding = torch.zeros(max_seq_len, embedding_dim)
    
    sinusoidal_embedding[:, 0::2] = torch.sin(position / denominators)
    sinusoidal_embedding[:, 1::2] = torch.cos(position / denominators)
    
    return sinusoidal_embedding
