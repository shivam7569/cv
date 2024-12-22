import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.utils.layers import DropPath
from cv.attention.transformers import ViTEncoder

class ViT(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Model Class for Vision Transformer architecture from paper on: An Image is Worth 16x16 Words"

    def __init__(
            self,
            num_classes,
            d_model,
            image_size,
            patch_size,
            classifier_mlp_d,
            encoder_mlp_d,
            encoder_num_heads,
            num_encoder_blocks,
            registers=None,
            dropout=0.0,
            encoder_dropout=0.0,
            encoder_attention_dropout=0.0,
            encoder_projection_dropout=0.0,
            patchify_technique="linear",
            stochastic_depth=False,
            stochastic_depth_mp=None,
            layer_scale=None,
            ln_order="residual",
            in_channels=3,
            classifier=True
        ):
        
        super(ViT, self).__init__()

        ViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        scale = d_model ** -0.5
        num_patches = (image_size // patch_size) ** 2

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.registers = registers
        self.d_model = d_model
        self.num_encoder_blocks = num_encoder_blocks
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify
        if patchify_technique == "convolutional":
            self.conv_patcher = nn.Conv2d(
            in_channels=in_channels, out_channels=d_model,
            kernel_size=patch_size, stride=patch_size, padding="valid"
        )

        self.linear_projection = nn.Linear(in_features=self.embed_size, out_features=d_model) if patchify_technique == "linear" else nn.Identity()

        self.class_token = nn.Parameter(
            scale * torch.rand((1, 1, d_model)), requires_grad=True
        )
        if registers is not None:
            self.register_tokens = nn.Parameter(
                torch.randn(1, registers, d_model)
            )

        self.position_embeddings = nn.Embedding(num_patches + 1, d_model)
        self.register_buffer(
            "position_ids",
            torch.arange(num_patches + 1).expand(1, -1),
            persistent=False
        )       

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout,
            stodepth=stochastic_depth, stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        ) if classifier else nn.Identity()

        self.final_ln = nn.LayerNorm(normalized_shape=d_model)

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
        conv_projection = self.conv_patcher(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2)

        return conv_projection

    def _initialize(self):
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5
        proj_std = (self.d_model ** -0.5) * ((2 * self.num_encoder_blocks) ** -0.5)

        for block in self.encoder.encoder_layers:
            nn.init.normal_(block.msa.w_q.weight, std=attn_std)
            nn.init.normal_(block.msa.w_k.weight, std=attn_std)
            nn.init.normal_(block.msa.w_v.weight, std=attn_std)
            nn.init.normal_(block.msa.w_o.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[-1].weight, std=proj_std)

        nn.init.normal_(tensor=self.position_embeddings.weight, mean=0.0, std=0.01)

    def updateStochasticDepthRate(self, k=0.05):
        for module_name, encoder_module in self.encoder.named_modules():
            if isinstance(encoder_module, DropPath):
                encoder_number = int(module_name.split(".")[1])
                encoder_module.drop_prob = encoder_module.drop_prob + encoder_number * (k / (len(self.encoder.encoder) - 1))

    def forward(self, x):
        x = self.patchify(x)
        x = self.linear_projection(x)

        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.position_embeddings(self.position_ids)
        x = self.dropout(x)

        if self.registers is not None:
            x = torch.cat([self.register_tokens.expand(x.size(0), -1, -1), x], dim=1)
            x = self.encoder(x)
            _, x = torch.split(x, [self.registers, x.size(1) - self.registers], dim=1)
        else:
            x = self.encoder(x)

        class_token = self.final_ln(x[:, 0, :])

        x = self.classifier(class_token)

        return x
