import math
import torch
import numpy as np
import torch.nn as nn

from cv.utils import MetaWrapper
from cv.utils.layers import DropPath, LayerScale
from cv.attention.variants import WindowAttention

class SwinTransformer(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Shifted Windows Transformer Block from paper on: Hierarchical Vision Transformer using Shifted Windows"

    def __init__(
            self,
            embed_c,
            d_ff_ratio,
            num_heads,
            window_size,
            shift,
            encoder_dropout,
            attention_dropout,
            projection_dropout,
            stodepth_prob=0.0,
            layer_scale=None,
            qkv_bias=False
    ):
        super(SwinTransformer, self).__init__()

        self.window_size = window_size
        self.shift = shift

        self.ln_1 = nn.LayerNorm(normalized_shape=embed_c)
        self.ln_2 = nn.LayerNorm(normalized_shape=embed_c)
        self.window_msa = WindowAttention(
            embed_dim=embed_c, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attention_dropout=attention_dropout,
            projection_dropout=projection_dropout, shift=shift
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_c, out_features=embed_c*d_ff_ratio),
            nn.GELU(),
            nn.Dropout(p=encoder_dropout) if encoder_dropout > 0.0 else nn.Identity(),
            nn.Linear(in_features=embed_c*d_ff_ratio, out_features=embed_c)
        )

        self.dropPath_1 = DropPath(
            drop_prob=stodepth_prob
        ) if stodepth_prob > 0.0 else nn.Identity()

        self.dropPath_2 = DropPath(
            drop_prob=stodepth_prob
        ) if stodepth_prob > 0.0 else nn.Identity()

        self.ls1 = LayerScale(num_channels=embed_c, init_value=layer_scale, type_="msa") if layer_scale is not None else nn.Identity()        
        self.ls2 = LayerScale(num_channels=embed_c, init_value=layer_scale, type_="msa") if layer_scale is not None else nn.Identity()

    def _to_windows(self, features):
        
        _, H, W, C = features.shape

        windows = []
        
        for i in range(H // self.window_size):
            for j in range(W // self.window_size):
                window = features[
                    :,
                    i * self.window_size: (i+1) * self.window_size,
                    j * self.window_size: (j+1) * self.window_size,
                    :
                ]
                windows.append(window)

        windows = torch.stack(windows, dim=1).view(-1, self.window_size, self.window_size, C)
        
        return windows
    
    def _from_windows(self, windows, batch_size, feature_h, feature_w):

        num_windows, _, _, C = windows.shape
        windows_per_dim = int(math.sqrt(num_windows // batch_size))

        windows = windows.view(batch_size, -1, self.window_size, self.window_size, C)
        features = torch.zeros(size=(batch_size, feature_h, feature_w, C), device=windows.device)

        for i in range(windows_per_dim):
            for j in range(windows_per_dim):
                window = windows[:, windows_per_dim*i+j, :, :, :]
                features[
                    :,
                    i * self.window_size: (i+1) * self.window_size,
                    j * self.window_size: (j+1) * self.window_size,
                    :
                ] = window

        return features
    
    def forward(self, x):
        
        B, H, W, C = x.shape

        if not hasattr(self, "attn_mask"):
            if self.shift > 0:
                Ht = torch.ceil(torch.tensor(H / self.window_size)) * self.window_size
                Wt = torch.ceil(torch.tensor(W / self.window_size)) * self.window_size
                img_mask = torch.zeros((1, int(Ht.item()), int(Wt.item()), 1))
                cnt = 0
                for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift),
                    slice(-self.shift, None)
                ):
                    for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift),
                        slice(-self.shift, None)
                    ):
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = self._to_windows(img_mask)
                mask_windows = mask_windows.view(-1, self.window_size**2)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(np.NINF)).masked_fill(attn_mask == 0, float(0.0)).to(x.device)
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask, persistent=False)

        if self.shift > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        else:
            shifted_x = x.clone()

        shifted_x = self._to_windows(shifted_x)
        shifted_x = shifted_x.view(-1, self.window_size**2, C)

        ln_1_out = self.ln_1(shifted_x)

        msa_out = self.window_msa(q=ln_1_out, k=ln_1_out, v=ln_1_out, mask=self.attn_mask)
        msa_out = msa_out.view(-1, self.window_size, self.window_size, C)
        msa_out = self._from_windows(msa_out, B, H, W)
        msa_out = self.dropPath_1(self.ls1(msa_out))

        if self.shift > 0:
            msa_out = torch.roll(msa_out, shifts=(self.shift, self.shift), dims=(1, 2))

        msa_residual = x + msa_out

        msa_residual = msa_residual.view(-1, self.window_size**2, C)

        ln_2_out = self.ln_2(msa_residual)

        mlp_out = self.mlp(ln_2_out)
        mlp_out = self.dropPath_2(self.ls2(mlp_out))

        mlp_residual = msa_residual + mlp_out

        mlp_residual = mlp_residual.view(B, H, W, C)

        return mlp_residual
