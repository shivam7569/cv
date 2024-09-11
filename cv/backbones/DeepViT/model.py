import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.utils.layers import DropPath
from cv.attention.transformers import ViTEncoder

class DeepViT(nn.Module, metaclass=MetaWrapper):
    """
    DeepViT: Vision Transformer architecture designed for enhanced depth and re-attention, 
    inspired by the `paper <https://arxiv.org/abs/2103.11886.pdf>`_.

    The DeepViT model is built to address the limitations of traditional Vision Transformers (ViTs) 
    when scaling depth. Standard ViTs struggle with gradient degradation and lack sufficient attention 
    at deeper layers, which can hinder performance in deeper architectures. DeepViT introduces several 
    innovations to overcome these challenges, including re-attention mechanisms, effective stochastic 
    depth regularization, and LayerScale initialization.

    Args:
        num_classes (int): Number of target classes for classification tasks.
        d_model (int): The dimensionality of the transformer embeddings (also referred to as the hidden size).
        image_size (int): Input image dimension (assumes a square image of size image_size x image_size).
        patch_size (int): Size of the square patches into which the image is divided.
        classifier_mlp_d (int): Dimensionality of the intermediate MLP in the classification head.
        encoder_mlp_d (int): Dimensionality of the feed-forward network within each transformer encoder block.
        encoder_num_heads (int): Number of attention heads in each multi-head self-attention (MHSA) layer.
        num_encoder_blocks (int): Number of transformer encoder layers (blocks) in the model.
        dropout (float, optional): Dropout probability applied after the linear projection and within the MLP layers (default: 0.0).
        encoder_dropout (float, optional): Dropout rate applied within the transformer encoder blocks (default: 0.0).
        encoder_attention_dropout (float, optional): Dropout probability applied to the attention layers (default: 0.0).
        encoder_projection_dropout (float, optional): Dropout applied to projections inside the transformer blocks (default: 0.0).
        patchify_technique (str, optional): Method used to divide the input image into patches. Options are "linear" for unfolding and "convolutional" for using convolution (default: "linear").
        stochastic_depth (bool, optional): Whether to use stochastic depth regularization (default: False).
        stochastic_depth_mp (float, optional): Maximum probability for stochastic depth, controlling the likelihood of dropping layers during training (default: None).
        layer_scale (float, optional): Scaling factor for LayerScale initialization. If None, LayerScale is disabled (default: None).
        ln_order (str, optional): Order of layer normalization application. Defaults to applying normalization after the residual connection (default: "residual").
        re_attention (bool, optional): Whether to enable the re-attention mechanism to refine attention maps (default: False).
        in_channels (int, optional): Number of input channels in the image, typically 3 for RGB (default: 3).

    Example:
        >>> model = DeepViT(num_classes=1000, d_model=384, image_size=224, patch_size=16, classifier_mlp_d=2048, encoder_mlp_d=1152, encoder_num_heads=12, num_encoder_blocks=32)
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for DeepViT architecture from paper on: Towards Deeper Vision Transformer"

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
            dropout=0.0,
            encoder_dropout=0.0,
            encoder_attention_dropout=0.0,
            encoder_projection_dropout=0.0,
            patchify_technique="linear",
            stochastic_depth=False,
            stochastic_depth_mp=None,
            layer_scale=None,
            ln_order="residual",
            re_attention=False,
            in_channels=3
        ):
        
        super(DeepViT, self).__init__()

        DeepViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=d_model),
            nn.LayerNorm(normalized_shape=d_model)
        )

        self.class_token = nn.Parameter(
            torch.rand((1, 1, d_model)), requires_grad=True
        )

        self.position_embeddings = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, 1, d_model)), requires_grad=True
        )
        nn.init.trunc_normal_(tensor=self.position_embeddings, mean=0.0, std=0.02)
        nn.init.trunc_normal_(tensor=self.class_token, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout,
            stodepth=stochastic_depth, stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order, re_attention=re_attention
        )
        self.stochastic_depth_mp = stochastic_depth_mp

        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

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
            in_channels=self.in_channels, out_channels=self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2)

        return conv_projection

    def updateStochasticDepthRate(self, k=0.05):
        """
        Updates the stochastic depth rate for each block in the transformer encoder.

        Stochastic depth is a regularization technique that randomly drops entire layers 
        during training to prevent overfitting. This method increases the drop probability 
        for each transformer encoder block based on its position in the model, using the 
        following formula:

        .. math::

            \\text{new_drop_prob} = \\text{original_drop_prob} + \\text{block_index} \\times \left( \\frac{k}{\\text{num_blocks} - 1} \\right)

        Args:
            k (float, optional): A scaling factor for adjusting the drop probability, default is 0.05. This value is spread across the transformer blocks, increasing progressively as you move deeper into the encoder.

        Example:
            If the model has 12 encoder blocks, and k=0.05, the first block will have its drop probability
            increased slightly, while the last block will have a larger increase, making the depth randomness
            more aggressive in the deeper layers.
        """

        for module_name, encoder_module in self.encoder.named_modules():
            if isinstance(encoder_module, DropPath):
                encoder_number = int(module_name.split(".")[1])
                encoder_module.drop_prob = encoder_module.drop_prob + encoder_number * (k / (len(self.encoder.encoder) - 1))

    def forward(self, x):
        """
        Forward pass of the DeepViT model. Processes the input image tensor through patchification, 
        linear projection, transformer encoder blocks, and classification, producing logits for the 
        target classes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        x = self.patchify(x)
        x = self.linear_projection(x)

        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
        x = self.dropout(x)

        x = self.encoder(x)

        class_token = x[:, 0, :]

        x = self.classifier(class_token)

        return x
