import torch
import torch.nn as nn

from cv.utils import MetaWrapper
from cv.attention.transformers import ViTEncoder, TransformerBlock

class CeiT(nn.Module, metaclass=MetaWrapper):

    """
    The `CeiT` class implements the CeiT (Convolutional Enhanced Image Transformer) architecture,
    as described in the `paper <https://arxiv.org/abs/2103.11816.pdf>`_.

    This model combines convolutional layers with transformer-based image processing, providing 
    a powerful approach for image classification tasks.

    Args:
        image_size (int, optional): The size of the input image (height and width) (default: 224).
        d_model (int, optional): The dimensionality of the model's feature space (default: 768).
        patch_size (int, optional): The size of the image patches to be extracted (default: 4).
        dropout (float, optional): The dropout rate applied to the inputs and outputs of the dropout layers (default: 0.0).
        encoder_num_heads (int, optional): Number of attention heads in the encoder layers (default: 12).
        num_encoder_blocks (int, optional): Number of encoder blocks in the transformer (default: 12).
        encoder_dropout (float, optional): Dropout rate applied in the encoder layers (default: 0.0).
        encoder_attention_dropout (float, optional): Dropout rate applied to the attention weights in the encoder (default: 0.0).
        encoder_projection_dropout (float, optional): Dropout rate applied to the projection layers in the encoder (default: 0.0).
        classifier_mlp_d (int, optional): Dimensionality of the hidden layer in the classifier's MLP (default: 2048).
        i2t_out_channels (int, optional): Number of output channels in the initial convolutional layer (default: 32).
        i2t_conv_kernel_size (int, optional): Size of the kernel in the initial convolutional layer (default: 7).
        i2t_conv_stride (int, optional): Stride of the convolution in the initial layer (default: 2).
        i2t_max_pool_kernel_size (int, optional): Kernel size of the max pooling layer (default: 3).
        i2t_max_pool_stride (int, optional): Stride of the max pooling layer (default: 2).
        leff_expand_ratio (int, optional): Expansion ratio for the Locally Enhanced Feed Forward (LEFF) module (default: 4).
        leff_depthwise_kernel (int, optional): Kernel size for the depthwise convolution in LEFF (default: 3).
        leff_depthwise_stride (int, optional): Stride for the depthwise convolution in LEFF (default: 1).
        leff_depthwise_padding (int, optional): Padding for the depthwise convolution in LEFF (default: 1).
        leff_depthwise_separable (bool, optional): Whether to use depthwise separable convolutions in LEFF (default: True).
        lca_encoder_expansion_ratio (int, optional): Expansion ratio for the LCA encoder module (default: 4).
        lca_encoder_num_heads (int, optional): Number of attention heads in the LCA encoder (default: 12).
        lca_encoder_dropout (float, optional): Dropout rate in the LCA encoder (default: 0.0).
        lca_encoder_attention_dropout (float, optional): Dropout rate for attention weights in the LCA encoder (default: 0.0).
        lca_encoder_projection_dropout (float, optional): Dropout rate for projection layers in the LCA encoder (default: 0.0).
        lca_encoder_ln_order (str, optional): The order of Layer Normalization in the LCA encoder (default: "post").
        lca_encoder_stodepth_prob (float, optional): Probability for stochastic depth in the LCA encoder (default: 0.0).
        lca_encoder_layer_scale (float, optional): Layer scaling factor in the LCA encoder (default: None).
        lca_encoder_qkv_bias (bool, optional): Whether to use bias in the QKV projections in the LCA encoder (default: False).
        patchify_technique (str, optional): Technique used for patch extraction ("linear" or "conv") (default: linear).
        stochastic_depth (bool, optional): Whether to use stochastic depth in the model (default: False).
        stochastic_depth_mp (float, optional): Stochastic depth max probability (default: None).
        layer_scale (float, optional): Scaling factor for the layers (default: None).
        ln_order (str, optional): The order of Layer Normalization in the model (default: "post").
        num_classes (int, optional): Number of classes for the classification task (default: 1000).
        in_channels (int, optional): Number of input channels in the images, typically 3 for RGB (default: 3).

    Example:
        >>> model = CeiT()
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for CeiT architecture from paper on: Incorporating Convolution Designs into Visual Transformers"

    def __init__(
            self,
            image_size=224,
            d_model=768,
            patch_size=4,
            dropout=0.0,
            encoder_num_heads=12,
            num_encoder_blocks=12,
            encoder_dropout=0.0,
            encoder_attention_dropout=0.0,
            encoder_projection_dropout=0.0,
            classifier_mlp_d=2048,
            i2t_out_channels=32,
            i2t_conv_kernel_size=7,
            i2t_conv_stride=2,
            i2t_max_pool_kernel_size=3,
            i2t_max_pool_stride=2,
            leff_expand_ratio=4,
            leff_depthwise_kernel=3,
            leff_depthwise_stride=1,
            leff_depthwise_padding=1,
            leff_depthwise_separable=True,
            lca_encoder_expansion_ratio=4,
            lca_encoder_num_heads=12,
            lca_encoder_dropout=0.0,
            lca_encoder_attention_dropout=0.0,
            lca_encoder_projection_dropout=0.0,
            lca_encoder_ln_order="post",
            lca_encoder_stodepth_prob=0.0,
            lca_encoder_layer_scale=None,
            lca_encoder_qkv_bias=False,
            patchify_technique="linear",
            stochastic_depth=False,
            stochastic_depth_mp=None,
            layer_scale=None,
            ln_order="post",
            num_classes=1000,
            in_channels=3
    ):

        super(CeiT, self).__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.i2t_out_channels = i2t_out_channels
        self.embed_size = (patch_size ** 2) * i2t_out_channels

        self.i2t_module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=i2t_out_channels,
                kernel_size=i2t_conv_kernel_size,
                stride=i2t_conv_stride
            ),
            nn.BatchNorm2d(num_features=i2t_out_channels),
            nn.MaxPool2d(
                kernel_size=i2t_max_pool_kernel_size,
                stride=i2t_max_pool_stride
            )
        )

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

        leff_params = {
            "num_tokens": (((((((image_size - i2t_conv_kernel_size) // i2t_conv_stride) + 1) - i2t_max_pool_kernel_size) // i2t_max_pool_stride) + 1) // patch_size) ** 2 + 1,
            "in_features": d_model,
            "expand_ratio": leff_expand_ratio,
            "depthwise_kernel": leff_depthwise_kernel,
            "depthwise_stride": leff_depthwise_stride,
            "depthwise_padding": leff_depthwise_padding,
            "depthwise_separable": leff_depthwise_separable
        }

        self.encoder = ViTEncoder(
            embed_dim=d_model, d_ff=None, num_heads=encoder_num_heads,
            num_blocks=num_encoder_blocks, encoder_dropout=encoder_dropout, 
            attention_dropout=encoder_attention_dropout, projection_dropout=encoder_projection_dropout,
            stodepth=stochastic_depth, stodepth_mp=stochastic_depth_mp, layer_scale=layer_scale, ln_order=ln_order, leff_params=leff_params
        )

        self.lca_attention = TransformerBlock(
            embed_dim=d_model, d_ff=d_model * lca_encoder_expansion_ratio, num_heads=lca_encoder_num_heads,
            encoder_dropout=lca_encoder_dropout, attention_dropout=lca_encoder_attention_dropout,
            projection_dropout=lca_encoder_projection_dropout, ln_order=lca_encoder_ln_order,
            stodepth_prob=lca_encoder_stodepth_prob, layer_scale=lca_encoder_layer_scale, qkv_bias=lca_encoder_qkv_bias, ceit_cross_attention=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=classifier_mlp_d),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=classifier_mlp_d, out_features=num_classes)
        )

    def _linear_patchify(self, img):
        linear_projection = nn.functional.unfold(
            input=img, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )
        linear_projection = linear_projection.transpose(dim0=1, dim1=2)

        return linear_projection

    def _conv_patchify(self, img):
        conv_projection = nn.Conv2d(
            in_channels=self.i2t_out_channels, out_channels=self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )(img)
        conv_projection = torch.flatten(input=conv_projection, start_dim=2)
        conv_projection = conv_projection.transpose(dim0=1, dim1=2)

        return conv_projection

    def forward(self, x):

        """
        Forward pass through the CeiT model.

        This method performs a forward pass of the input tensor through the CeiT model. It includes the following steps:

        1. Initial feature extraction using the `i2t_module` (i2t: image-to-tokens).
        2. Transformation of the extracted features into patches using the `patchify` method.
        3. Linear projection of the patchified features.
        4. Addition of positional embeddings and class tokens.
        5. Application of dropout.
        6. Processing of the features through the transformer encoder.
        7. Application of LCA attention to the encoder's output.
        8. Classification of the final token output using the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the model, with shape (batch_size, num_classes).

        Notes:
            - The `i2t_module` first processes the input images, extracting initial features through convolutional layers followed by max pooling.
            - The `patchify` method converts the feature maps into a sequence of patches suitable for the transformer encoder.
            - Positional embeddings and class tokens are added to the sequence of patches to incorporate positional information and a class-specific token.
            - Dropout is applied to prevent overfitting during training.
            - The transformer encoder processes the sequence of patches, producing an encoded representation.
            - LCA attention is applied to the encoded representation to enhance the feature representation.
            - Finally, the class token is passed through a classifier to obtain the predicted class scores.

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))  # Example input tensor of shape (batch_size, channels, height, width)
        """

        x = self.i2t_module(x)
        x = self.patchify(x)
        x = self.linear_projection(x)

        x = torch.cat([self.class_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.position_embeddings.expand(x.size(0), x.size(1), -1)
        x = self.dropout(x)

        lca_class_tokens = torch.stack(self.encoder(x), dim=1)
        class_token = self.lca_attention(lca_class_tokens).squeeze(1)

        out = self.classifier(class_token)

        return out
