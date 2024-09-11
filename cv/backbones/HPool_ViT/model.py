import torch
import torch.nn as nn

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.utils.layers import DropPath
from cv.attention.transformers import ViTEncoder

class HPool_ViT(nn.Module, metaclass=MetaWrapper):

    """
    HPool-ViT model implementing the Vision Transformer with Hierarchical Pooling (HPool) from `paper <https://arxiv.org/abs/2103.10619.pdf>`_.

    This model is based on the architecture introduced in the paper "Scalable Vision Transformers with Hierarchical Pooling".
    HPool-ViT uses a hierarchical pooling mechanism to improve scalability and efficiency in processing large images,
    while maintaining the core components of a Vision Transformer (ViT) such as patch embeddings, multi-head attention, 
    and Transformer encoders.

    Args:
        num_classes (int): Number of output classes for classification.
        d_model (int): Dimension of the model (embedding size for patches).
        image_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each image patch (assumed to be square).
        classifier_mlp_d (int): Dimension of the MLP in the classifier head.
        encoder_mlp_d (int): Dimension of the MLP in the Transformer encoder.
        encoder_num_heads (int): Number of attention heads in the Transformer encoder.
        num_encoder_blocks (int): Number of Transformer encoder blocks.
        dropout (float, optional): Dropout rate applied to the patch embeddings and classifier (default: 0.0).
        encoder_dropout (float, optional): Dropout rate in the encoder layers (default: 0.0).
        encoder_attention_dropout (float, optional): Dropout rate in the multi-head attention layers (default: 0.0).
        encoder_projection_dropout (float, optional): Dropout rate for the linear projections in the encoder (default: 0.0).
        patchify_technique (str, optional): Technique for creating patches from the image. Can be "linear" or "convolutional" (default: "linear").
        stochastic_depth (bool, optional): Whether to apply stochastic depth (DropPath) to encoder layers (default: False).
        stochastic_depth_mp (float or None): Maximum probability for stochastic depth. If None, no stochastic depth is applied (default: None).
        layer_scale (float or None): Scale factor for layer normalization. If None, no scaling is applied (default: None).
        ln_order (str, optional): Order of layer normalization. Can be "residual" or "pre" (default: "residual").
        hvt_pool (list of int or None): Transformer blocks to use hierarchical pooling at. If None, hierarchical pooling is not used (default: [1, 5, 9]).
        in_channels (int, optional): Number of input channels, typically 3 for RGB (default: 3).

    Example:
        >>> model = HPool_ViT(num_classes=1000, d_model=768, image_size=192, patch_size=16, classifier_mlp_d=2048, encoder_mlp_d=3072, encoder_num_heads=12, num_encoder_blocks=12)

    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for HPool-ViT architecture from paper on: Scalable Vision Transformers with Hierarchical Pooling"

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
            hvt_pool=[1, 5, 9],
            in_channels=3
        ):
        
        super(HPool_ViT, self).__init__()

        HPool_ViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=d_model),
            nn.LayerNorm(normalized_shape=d_model)
        )

        self.position_embeddings = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(1, 1, d_model)), requires_grad=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout, stodepth=stochastic_depth, 
            stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order, hvt_pool=hvt_pool
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
        Forward pass through the HPool-ViT model.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, in_channels, height, width)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, num_classes)` containing the predicted class scores.
        
        Example:
            >>> output = model(torch.randn(1, 3, 192, 192))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        x = self.patchify(x)
        x = self.linear_projection(x)

        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
        x = self.dropout(x)

        x = self.encoder(x)

        class_token_or_hvt = x[:, 0, :]

        x = self.classifier(class_token_or_hvt)

        return x
