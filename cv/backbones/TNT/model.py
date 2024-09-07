import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.attention.transformers import TNTEncoder

class TNT(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for Transformer in Transformer architecture from paper on: Transformer in Transformer"

    def __init__(
            self,
            image_size=224,
            patch_size=16,
            pixel_size=4,
            patch_embed=640,
            pixel_embed=40,
            patch_d_ff=2560,
            pixel_d_ff=160,
            patch_num_heads=10,
            pixel_num_heads=4,
            num_blocks=12,
            patch_encoder_dropout=0.0,
            patch_attention_dropout=0.0,
            patch_projection_dropout=0.0,
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
            pixel_projection_dropout=0.0,
            pixel_ln_order="pre",
            pixel_stodepth=False,
            pixel_stodepth_mp=None,
            pixel_layer_scale=None,
            pixel_se_block=None,
            pixel_se_points="both",
            pixel_qkv_bias=False,
            pixel_in_dims=None,
            patchify_technique="linear",
            tnt_dropout=0.0,
            classifier_mlp_d=2048,
            classifier_dropout=0.0,
            num_classes=1000,
            in_channels=3
            ):

        super(TNT, self).__init__()

        TNT._assertions(image_size=image_size, patch_size=patch_size, pixel_size=pixel_size, patchify_technique=patchify_technique)

        self.patch_size = patch_size
        self.pixel_size = pixel_size
        self.in_channels = in_channels

        sentence_embed_dim = (patch_size ** 2) * in_channels
        word_embed_dim = (pixel_size ** 2) * in_channels

        num_sentences_in_image = (image_size // self.patch_size) ** 2
        num_words_in_sentence = (self.patch_size // self.pixel_size) ** 2

        self.patch_position_encodings = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, num_sentences_in_image + 1, patch_embed), mean=0.0, std=0.02), requires_grad=True
        )
        self.pixel_position_encodings = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, num_words_in_sentence, pixel_embed), mean=0.0, std=0.02), requires_grad=True
        )

        self.sentence_class_token = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, 1, patch_embed), mean=0.0, std=0.02), requires_grad=True
        )

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection_sentences = nn.Sequential(
            nn.Linear(in_features=sentence_embed_dim, out_features=patch_embed),
            nn.LayerNorm(normalized_shape=patch_embed)
        )
        self.linear_projection_words = nn.Sequential(
            nn.Linear(in_features=word_embed_dim, out_features=pixel_embed),
            nn.LayerNorm(normalized_shape=pixel_embed)
        )

        self.tnt_dropout = nn.Dropout(p=tnt_dropout)

        self.tnt_encoder = TNTEncoder(
            patch_embed_dim=patch_embed,
            patch_d_ff=patch_d_ff,
            patch_num_heads=patch_num_heads,
            pixel_embed_dim=pixel_embed,
            pixel_d_ff=pixel_d_ff,
            pixel_num_heads=pixel_num_heads,
            num_blocks=num_blocks,
            patch_encoder_dropout=patch_encoder_dropout,
            patch_attention_dropout=patch_attention_dropout,
            patch_projection_dropout=patch_projection_dropout,
            patch_ln_order=patch_ln_order,
            patch_stodepth=patch_stodepth,
            patch_stodepth_mp=patch_stodepth_mp,
            patch_layer_scale=patch_layer_scale,
            patch_se_block=patch_se_block,
            patch_se_points=patch_se_points,
            patch_qkv_bias=patch_qkv_bias,
            patch_in_dims=patch_in_dims,
            pixel_encoder_dropout=pixel_encoder_dropout,
            pixel_attention_dropout=pixel_attention_dropout,
            pixel_projection_dropout=pixel_projection_dropout,
            pixel_ln_order=pixel_ln_order,
            pixel_stodepth=pixel_stodepth,
            pixel_stodepth_mp=pixel_stodepth_mp,
            pixel_layer_scale=pixel_layer_scale,
            pixel_se_block=pixel_se_block,
            pixel_se_points=pixel_se_points,
            pixel_qkv_bias=pixel_qkv_bias,
            pixel_in_dims=pixel_in_dims
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=patch_embed, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

    @staticmethod
    def _assertions(image_size, patch_size, pixel_size, patchify_technique):
        assert image_size % patch_size == 0, Global.LOGGER.error(f"Patch size {patch_size} is not a divisor of image dimension {image_size}")
        assert patch_size % pixel_size == 0, Global.LOGGER.error(f"Pixel size {patch_size} is not a divisor of patch dimension {image_size}")
        assert patchify_technique in ("linear", "convolutional"), Global.LOGGER.error(f"{patchify_technique} patchify technique not supported")

    def _linear_patchify(self, img, patch_size):
        linear_projection = nn.functional.unfold(
            input=img, kernel_size=(patch_size, patch_size), stride=patch_size
        )
        linear_projection = linear_projection.transpose(dim0=1, dim1=2)

        return linear_projection

    def _conv_patchify(self, img, patch_size):
        conv_projection = nn.Conv2d(
            in_channels=self.in_channels, out_channels=(patch_size ** 2) * self.in_channels,
            kernel_size=patch_size, stride=patch_size
        )(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2)

        return conv_projection
    
    def forward(self, x):
        sentences = self.patchify(x, patch_size=self.patch_size)
        words = self.patchify(
            sentences.reshape(
                -1, self.in_channels, self.patch_size, self.patch_size
            ), patch_size=self.pixel_size
        )

        projected_sentences = self.linear_projection_sentences(sentences)
        projected_words = self.linear_projection_words(words)

        outer_transformer_in = torch.cat(
            [
                self.sentence_class_token.expand(projected_sentences.size(0), -1, -1),
                projected_sentences
            ], dim=1
        ) + self.patch_position_encodings.expand(projected_sentences.size(0), -1, -1)

        inner_transformer_in = projected_words + self.pixel_position_encodings.expand(projected_words.size(0), -1, -1)

        inner_transformer_in = self.tnt_dropout(inner_transformer_in)
        outer_transformer_in = self.tnt_dropout(outer_transformer_in)

        sentences = self.tnt_encoder(outer_transformer_in, inner_transformer_in)

        class_token = sentences[:, 0, :]
        logits = self.classifier(class_token)

        return logits
