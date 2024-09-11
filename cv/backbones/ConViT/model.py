import torch
import torch.nn as nn
from cv.attention.transformers.vit_encoder import ViTEncoder

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.utils.layers import DropPath

class ConViT(nn.Module, metaclass=MetaWrapper):

    """
    Model class for `ConViT` architecture as described in the `paper <https://arxiv.org/abs/2103.10697.pdf>`_.

    ConViT is a hybrid architecture that combines vision transformers (ViT) with convolutional
    inductive biases, enabling the model to better capture local features in images. This is
    achieved through gated transformer blocks, allowing flexibility in balancing local and global feature extraction.

    Args:
        d_model (int): Dimension of the model (embedding size).
        image_size (int): Height and width of the input image.
        patch_size (int): Size of the patches extracted from the image.
        classifier_mlp_d (int): Hidden size of the classifier MLP layer.
        encoder_mlp_d (int): Hidden size of the MLP within the transformer encoder.
        encoder_num_heads (int): Number of attention heads in each transformer encoder block.
        num_encoder_blocks (int): Total number of encoder blocks.
        num_gated_blocks (int, optional): Number of gated transformer blocks (default: 10).
        locality_strength (float, optional): Strength of the locality bias in gated blocks (default: 1.0).
        locality_distance_method (str, optional): Method for determining locality distance (default: "constant").
        use_conv_init (bool, optional): Whether to use convolutional initialization (default: True).
        d_pos (int, optional): Dimensionality of positional embeddings (default: 3).
        dropout (float, optional): Dropout probability for regularization (default: 0.0).
        encoder_dropout (float, optional): Dropout probability within the encoder (default: 0.0).
        encoder_attention_dropout (float, optional): Dropout probability in attention layers (default: 0.0).
        encoder_projection_dropout (float, optional): Dropout probability in projection layers (default: 0.0).
        patchify_technique (str, optional): Patchification technique, either "linear" or "convolutional" (default: "linear").
        stochastic_depth (bool, optional): Whether to use stochastic depth (default: False).
        stochastic_depth_mp (optional): Stochastic depth max probability (default: None).
        layer_scale (optional): Scale factor for layer normalization (default: None).
        ln_order (str, optional): Layer normalization order (default: "residual").
        in_channels (int, optional): Number of input channels, typically 3 for RGB images (default: 3).
        num_classes (int, optional): Number of output classes (default: 1000).

    Example:
        >>> model = ConViT(d_model=8, image_size=224, patch_size=16, classifier_mlp_d=2048, encoder_mlp_d=4096, encoder_num_heads=16, num_encoder_blocks=12)
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for ConViT architecture from paper on: Improving Vision Transformers with Soft Convolutional Inductive Biases"

    def __init__(
            self,
            d_model,
            image_size,
            patch_size,
            classifier_mlp_d,
            encoder_mlp_d,
            encoder_num_heads,
            num_encoder_blocks,
            num_gated_blocks=10,
            locality_strength=1.0,
            locality_distance_method="constant",
            use_conv_init=True,
            d_pos=3,
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
            num_classes=1000
        ):
        
        super(ConViT, self).__init__()

        ConViT._assertions(image_size=image_size, patch_size=patch_size, patchify_technique=patchify_technique)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_size = (patch_size ** 2) * in_channels

        self.patchify = self._linear_patchify if patchify_technique == "linear" else self._conv_patchify

        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=d_model),
            nn.LayerNorm(normalized_shape=d_model)
        )

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=encoder_mlp_d, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout,
            stodepth=stochastic_depth, stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order,
            hvt_pool=None, gated_transformer_params={
                "locality_strength": locality_strength,
                "locality_distance_method": locality_distance_method,
                "use_conv_init": use_conv_init,
                "d_pos": d_pos,
                "num_blocks": num_gated_blocks
            }
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
        Forward pass of the ConViT model.

        The input tensor `x` is first patchified, then linearly projected into the transformer space.
        It is passed through a transformer encoder with both global and gated blocks to capture local
        and global features. The final class token is used for classification.

        Args:
            x (torch.Tensor): The input tensor representing a batch of images, with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The classification output with shape (batch_size, num_classes), representing the predicted class probabilities for each image.

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        x = self.patchify(x)
        x = self.linear_projection(x)
        x = self.dropout(x)
        x = self.encoder(x)

        class_token = x[:, 0, :]

        x = self.classifier(class_token)

        return x
