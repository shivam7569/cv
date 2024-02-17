import torch.nn as nn

from attention.transformers.vanilla_transformer import TransformerBlock

class TNTEncoder(nn.Module):

    def __init__(
            self,
            patch_embed_dim,
            patch_d_ff,
            patch_num_heads,
            pixel_embed_dim,
            pixel_d_ff,
            pixel_num_heads,
            num_blocks,
            patch_encoder_dropout=0.0,
            patch_attention_dropout=0.0,
            patch_ln_order="pre",
            patch_stodepth=False,
            patch_stodepth_mp=None,
            patch_layer_scale=None,
            patch_se_block=None,
            patch_se_points="both",
            patch_qkv_bias=False,
            patch_in_dims=None,
            pixel_encoder_dropout=0.0,
            pixel_attention_dropout=0.0,
            pixel_ln_order="pre",
            pixel_stodepth=False,
            pixel_stodepth_mp=None,
            pixel_layer_scale=None,
            pixel_se_block=None,
            pixel_se_points="both",
            pixel_qkv_bias=False,
            pixel_in_dims=None
        ):

        super(TNTEncoder, self).__init__()

        self.num_blocks = num_blocks

        if patch_stodepth:
            patch_drop_probs = [patch_stodepth_mp / (num_blocks - 1) * i for i in range(num_blocks)]
        else:
            patch_drop_probs = [0.0 for _ in range(num_blocks)]

        self.outer_transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=patch_embed_dim, d_ff=patch_d_ff, num_heads=patch_num_heads,
                    encoder_dropout=patch_encoder_dropout, attention_dropout=patch_attention_dropout,
                    ln_order=patch_ln_order, stodepth_prob=patch_drop_probs[i], layer_scale=patch_layer_scale,
                    se_block=patch_se_block, se_points=patch_se_points, qkv_bias=patch_qkv_bias, in_dims=patch_in_dims
                ) for i in range(num_blocks)
            ]
        )

        if pixel_stodepth:
            pixel_drop_probs = [pixel_stodepth_mp / (num_blocks - 1) * i for i in range(num_blocks)]
        else:
            pixel_drop_probs = [0.0 for _ in range(num_blocks)]

        self.inner_transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=pixel_embed_dim, d_ff=pixel_d_ff, num_heads=pixel_num_heads,
                    encoder_dropout=pixel_encoder_dropout, attention_dropout=pixel_attention_dropout,
                    ln_order=pixel_ln_order, stodepth_prob=pixel_drop_probs[i], layer_scale=pixel_layer_scale,
                    se_block=pixel_se_block, se_points=pixel_se_points, qkv_bias=pixel_qkv_bias, in_dims=pixel_in_dims
                ) for i in range(num_blocks)
            ]
        )
        

    def forward(self, sentences, words):
        batch_size = sentences.size(0)
        num_sentences = sentences.size(1) - 1

        for i in range(self.num_blocks):
            outer_transformer = self.outer_transformer_blocks[i]
            inner_transformer = self.inner_transformer_blocks[i]

            words = inner_transformer(words)
            words_residual = nn.functional.pad(
                words.view(batch_size, num_sentences, -1), pad=(0, 0, 1, 0), mode="constant", value=0
            )

            sentences += words_residual
            sentences = outer_transformer(sentences)

        return sentences
