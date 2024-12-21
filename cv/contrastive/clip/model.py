import torch
import numpy as np
import torch.nn as nn
from functools import lru_cache, partial

from cv.backbones import ViT
from cv.utils import MetaWrapper
from cv.src.tokenizer import SimpleTokenizer, tokenize
from cv.attention.transformers import TransformerBlock


class Clip(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for CLIP architecture from paper on: Learning Transferable Visual Models From Natural Language Supervision"

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
            text_num_blocks,
            context_dim,
            vit_dropout,
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

        if stodepth:
            drop_probs = [stochastic_depth_mp / (text_num_blocks - 1) * i for i in range(text_num_blocks)]
        else:
            drop_probs = [0.0 for _ in range(text_num_blocks)]

        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.text_num_blocks = text_num_blocks

        self.vit = ViT(
            num_classes=None, d_model=embed_dim, image_size=image_size, patch_size=patch_size,
            classifier_mlp_d=None, encoder_mlp_d=d_ff, encoder_num_heads=vit_num_heads,
            num_encoder_blocks=vit_num_blocks, dropout=vit_dropout, encoder_dropout=encoder_dropout,
            encoder_attention_dropout=attention_dropout, encoder_projection_dropout=projection_dropout,
            patchify_technique=patchify_technique, stochastic_depth=stodepth, stochastic_depth_mp=stochastic_depth_mp,
            layer_scale=layer_scale, ln_order=ln_order, in_channels=in_channels, classifier=False
        )

        self.img_embed_proj = nn.Linear(in_features=embed_dim, out_features=proj_dim, bias=False)

        self.text_encoder = [
            TransformerBlock(
                embed_dim=embed_dim, d_ff=d_ff, num_heads=text_num_heads,
                encoder_dropout=encoder_dropout, attention_dropout=attention_dropout,
                projection_dropout=projection_dropout, ln_order=ln_order,
                stodepth_prob=drop_probs[i], layer_scale=layer_scale
            ) for i in range(text_num_blocks)
        ]
        
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

    def _initialize(self):
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        proj_std = (self.embed_dim ** -0.5) * ((2 * self.text_num_blocks) ** -0.5)

        nn.init.normal_(self.text_embeddings.weight, std=0.02)
        nn.init.normal_(self.txt_position_embedding.weight, std=0.01)
        nn.init.normal_(self.txt_embed_proj.weight, std=self.embed_dim ** -0.5)

        for block in self.text_encoder:
            nn.init.normal_(block.msa.w_q.weight, std=attn_std)
            nn.init.normal_(block.msa.w_k.weight, std=attn_std)
            nn.init.normal_(block.msa.w_v.weight, std=attn_std)
            nn.init.normal_(block.msa.w_o.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[-1].weight, std=proj_std)

    @lru_cache
    def _build_mask(self):
        mask = torch.ones(self.context_dim, self.context_dim)
        mask.triu_(1)

        return mask.bool()
    
    def _encode_img(self, img):
        img_embeddings = self.vit(img)
        img_embeddings = self.img_embed_proj(img_embeddings)

        return img_embeddings

    def _encode_txt(self, txt):
        tokenized_texts = self.text_tokenizer(txt)
        txt_tokens = self.text_embeddings(tokenized_texts)
        txt_tokens = txt_tokens + self.txt_position_embedding(self.txt_position_ids)

        for idx, encoder_block in enumerate(self.text_encoder):
            if idx == 0:
                txt_embeddings = encoder_block(txt_tokens, mask=self._build_mask())
            else:
                txt_embeddings = encoder_block(txt_embeddings, mask=self._build_mask())

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

clp = Clip(224, 16, 768, 512, 768*4, 12, 12, 8, 12, 77, 0.0, 0.0, 0.0, 0.0, 0.07, "pre", True, 0.1, 1e-6, "convolutional", 3)
x = torch.randn(2, 3, 224, 224)
t = ["a horse", "the cat"]
a, b = clp(x, t)

print(a)
print(b)