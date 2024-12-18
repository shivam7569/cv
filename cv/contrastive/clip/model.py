import torch
import numpy as np
import torch.nn as nn
from functools import partial

from cv.utils import Global
from cv.src.tokenizer import SimpleTokenizer, tokenize
from cv.attention.transformers import TransformerBlock


class Clip(nn.Module):

    def __init__(
            self,
            image_size,
            patch_size,
            embed_dim,
            proj_dim,
            d_ff,
            vit_num_heads,
            vit_num_blocks,
            text_num_heads,
            context_dim,
            encoder_dropout,
            attention_dropout,
            projection_dropout,
            temperature,
            ln_order="pre",
            stodepth=True,
            stochastic_depth_mp=0.1,
            layer_scale=1e-6,
            patchify_technique="convolutional",
            in_channels=3
    ):
        
        super(Clip, self).__init__()

        Clip._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.vit_num_blocks = vit_num_blocks

        scale = embed_dim ** -0.5
        num_patches = (image_size // patch_size) ** 2

        if stodepth:
            drop_probs = [stochastic_depth_mp / (vit_num_blocks - 1) * i for i in range(vit_num_blocks)]
        else:
            drop_probs = [0.0 for _ in range(vit_num_blocks)]
        
        self.image_encoder = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim, d_ff=d_ff, num_heads=vit_num_heads, encoder_dropout=encoder_dropout,
                attention_dropout=attention_dropout, projection_dropout=projection_dropout,
                stodepth_prob=drop_probs[i], ln_order=ln_order, layer_scale=layer_scale
            ) for i in range(vit_num_blocks)
        ])

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.class_token = nn.Parameter(
            scale * torch.rand((1, 1, embed_dim)), requires_grad=True
        )
        self.img_position_embedding = nn.Embedding(num_embeddings=num_patches + 1, embedding_dim=embed_dim)
        self.register_buffer(
            "img_position_ids",
            torch.arange(num_patches + 1).expand(1, -1),
            persistent=False
        )
        self.img_ln = nn.LayerNorm(normalized_shape=embed_dim)
        self.img_embed_proj = nn.Linear(in_features=embed_dim, out_features=proj_dim, bias=False)

        self.text_encoder = TransformerBlock(
            embed_dim=embed_dim, d_ff=d_ff, num_heads=text_num_heads, encoder_dropout=encoder_dropout,
            attention_dropout=attention_dropout, projection_dropout=projection_dropout,
            ln_order=ln_order, layer_scale=layer_scale
        )
        self.vocab_size = SimpleTokenizer().vocab_size
        self.text_tokenizer = partial(tokenize, context_length=context_dim, truncate=False)
        self.text_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_dim)
        self.txt_position_embedding = nn.Embedding(num_embeddings=context_dim, embedding_dim=embed_dim)
        self.register_buffer(
            "txt_position_ids",
            torch.arange(context_dim).expand(1, -1),
            persistent=False
        )
        self.txt_embed_proj = nn.Linear(in_features=embed_dim, out_features=proj_dim, bias=False)
        self.txt_ln = nn.LayerNorm(normalized_shape=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones(size=[]) * np.log(1 / temperature))

        self._initialize()

    @staticmethod
    def _assertions(image_size, patch_size, patchify_technique):
        assert image_size % patch_size == 0, Global.LOGGER.error(f"Patch size {patch_size} is not a divisor of image dimension {image_size}")
        assert patchify_technique in ("linear", "convolutional"), Global.LOGGER.error(f"{patchify_technique} patchify technique not supported")

    def _linear_patchify(self, img):
        linear_projection = nn.functional.unfold(
            input=img, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )
        linear_projection = linear_projection.transpose(dim0=1, dim1=2)

        return linear_projection

    def _conv_patchify(self, img):
        conv_projection = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size
        )(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2) # shape: (batch_size, num_patches, embedding_size)

        return conv_projection

    def _initialize(self):
        nn.init.normal_(self.text_embeddings.weight, std=0.02)
        nn.init.normal_(self.txt_position_embedding.weight, std=0.01)

        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        proj_std = (self.embed_dim ** -0.5) * ((2 * self.vit_num_blocks) ** -0.5)

        for block in self.image_encoder:
            nn.init.normal_(block.msa.w_q.weight, std=attn_std)
            nn.init.normal_(block.msa.w_k.weight, std=attn_std)
            nn.init.normal_(block.msa.w_v.weight, std=attn_std)
            nn.init.normal_(block.msa.w_o.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[-1].weight, std=proj_std)

        nn.init.normal_(self.txt_embed_proj.weight, std=attn_std)

    def _build_mask(self):
        mask = torch.ones(self.context_dim, self.context_dim)
        mask.triu_(1)

        return mask.bool()
    
    def _encode_img(self, img):
        img_tokens = self.patchify(img)
        img_tokens = torch.cat([self.class_token.expand(img.size(0), -1, -1), img_tokens], dim=1)
        img_tokens = img_tokens + self.img_position_embedding(self.img_position_ids)
        img_embeddings = self.image_encoder(img_tokens)
        img_embeddings = self.img_ln(img_embeddings[:, 0, :])
        img_embeddings = self.img_embed_proj(img_embeddings)

        return img_embeddings

    def _encode_txt(self, txt):
        tokenized_texts = self.text_tokenizer(txt)
        txt_tokens = self.text_embeddings(tokenized_texts)
        txt_tokens = txt_tokens + self.txt_position_embedding(self.txt_position_ids)
        txt_embeddings = self.text_encoder(txt_tokens, mask=self._build_mask())
        txt_embeddings = self.txt_ln(txt_embeddings)[
            torch.arange(txt_embeddings.shape[0]), tokenized_texts.argmax(dim=-1)
        ]
        txt_embeddings = self.txt_embed_proj(txt_embeddings)

        return txt_embeddings

    def forward(self, img, text):
        img_embeddings = self._encode_img(img=img)
        txt_embeddings = self._encode_txt(text)

        img_embeddings = img_embeddings / img_embeddings.norm(dim=1, keepdim=True)
        txt_embeddings = txt_embeddings / txt_embeddings.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_embeddings @ txt_embeddings.t()
        logits_per_text = logits_per_image.t()

        return (logits_per_image, logits_per_text)
