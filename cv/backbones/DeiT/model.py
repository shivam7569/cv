import torch
import torch.nn as nn

from cv import backbones
from cv.utils import Global
from cv.backbones import ViT
from cv.src.checkpoints import Checkpoint
from cv.utils import MetaWrapper

class DeiT(ViT, metaclass=MetaWrapper):

    """
    Data-efficient Image Transformer (DeiT) model class from `paper <https://arxiv.org/abs/2012.12877.pdf>`_.

    This class implements the DeiT architecture, which is designed to efficiently train Vision Transformers
    with the additional feature of knowledge distillation. DeiT uses a teacher-student framework where the 
    student model (DeiT) is trained to mimic the output of a pre-trained teacher model.

    Args:
        - num_classes (int): Number of output classes for classification.
        - d_model (int): Dimensionality of the model's hidden representations.
        - image_size (int): Size of the input image (should be divisible by patch_size).
        - patch_size (int): Size of the patches used in the image processing.
        - classifier_mlp_d (int): Dimensionality of the intermediate MLP in the classification head.
        - encoder_mlp_d (int): Dimension of the feed-forward layers in the Transformer encoder.
        - encoder_num_heads (int): Number of attention heads in the Transformer encoder.
        - num_encoder_blocks (int): Number of blocks in the Transformer encoder.
        - dropout (float, optional): Dropout probability applied after the linear projection and within the MLP layers.
        - encoder_dropout (float, optional): Dropout rate applied within the transformer encoder blocks.
        - encoder_attention_dropout (float, optional): Dropout probability applied to the attention layers.
        - patchify_technique (str, optional): Method used to divide the input image into patches. Options are "linear" for unfolding and "convolutional" for using convolution.
        - stochastic_depth (bool, optional): Whether to use stochastic depth regularization.
        - stochastic_depth_mp (float, optional): Maximum probability for stochastic depth, controlling the likelihood of dropping layers during training.
        - layer_scale (float, optional): Scaling factor for LayerScale initialization. If None, LayerScale is disabled.
        - return_logits_type (str): Type of logits to return. Options are "classification", "distillation", or "fusion".
        - teacher_model_name (str): Name of the pre-trained teacher model to use for distillation.
        - in_channels (int, optional): Number of input channels in the image, typically 3 for RGB.

    Note:
        Make sure the weights for the teacher model exists.

    Example:
        >>> model = DeiT(num_classes=1000, d_model=768, image_size=224, patch_size=16, classifier_mlp_d=2048, encoder_mlp_d=3072, encoder_num_heads=12, num_encoder_blocks=12, dropout=0.1, encoder_dropout=0.0, encoder_attention_dropout=0.0, patchify_technique="linear", stochastic_depth=False, stochastic_depth_mp=0.0, layer_scale=None, return_logits_type="fusion", teacher_model_name="ConvNeXt", in_channels=3)
    """

    @classmethod
    def __class_repr__(cls):
        return "Model Class for DeiT architecture from paper on: Training data-efficient image transformers & distillation through attention"

    def __init__(self, **kwargs):

        self.return_logits_type = kwargs["return_logits_type"]
        self.teacher_model_name = kwargs["teacher_model_name"]

        assert self.return_logits_type in ["classification", "distillation", "fusion"]

        kwargs.pop("return_logits_type")
        kwargs.pop("teacher_model_name")

        super(DeiT, self).__init__(**kwargs)

        if Global.LOGGER is not None: Global.LOGGER.info(f"Using {self.teacher_model_name} as teacher model")

        self.distillation_token = nn.Parameter(torch.rand(self.class_token.shape), requires_grad=True)
        self.teacher_model = getattr(backbones, self.teacher_model_name)()
        self.loadTeacherModel()

        self.classifier = nn.Linear(
            in_features=kwargs["d_model"], out_features=kwargs["num_classes"]
        )
        self.distillation_token_classifier = nn.Linear(
            in_features=kwargs["d_model"], out_features=kwargs["num_classes"]
        )

    def loadTeacherModel(self):
        """
        Loads and prepares the pre-trained teacher model for distillation.

        This method loads the teacher model from checkpoints, sets it to evaluation mode, and disables gradient computation.
        """

        self.teacher_model = Checkpoint.load(
            model=self.teacher_model, name=self.teacher_model_name
        )
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_model.eval()

    def student_model(self, x):
        """
        Performs a forward pass of the student model, including the handling of the distillation token.

        This method processes the input through the student model, computes logits for the class token and 
        optionally for the distillation token based on the `return_logits_type` setting.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, channels, height, width)`.

        Returns:
            Tensor: The output logits from the student model. The type of logits returned is determined by:
                - If `return_logits_type` is "classification": Returns logits from the classification head.
                - If `return_logits_type` is "distillation": Returns logits from the distillation token classifier.
                - If `return_logits_type` is "fusion": Returns the average of logits from both classifiers.
        """

        x = self.patchify(x)
        x = self.linear_projection(x)

        x = torch.cat(
            [
                self.class_token.expand(x.size(0), -1, -1),
                x,
                self.distillation_token.expand(x.size(0), -1, -1)
            ], dim=1
        )
        x += self.position_embeddings.expand(x.size(0), x.size(1), -1)
        x = self.dropout(x)

        x = self.encoder(x)

        class_token = x[:, 0, :]
        class_token_classifier = self.classifier(class_token)

        if self.return_logits_type == "classification": return class_token_classifier

        distillation_token = x[:, -1, :]
        distillation_token_classifier = self.distillation_token_classifier(distillation_token)

        if self.return_logits_type == "distillation": return distillation_token_classifier

        if self.return_logits_type == "fusion":
            return (class_token_classifier + distillation_token_classifier) / 2

    def forward(self, x):
        """
        Performs a forward pass of the DeiT model.

        This method computes the outputs of both the student and teacher models. It returns a tuple containing 
        the student model's output and the teacher model's output.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, channels, height, width)`.

        Returns:
            tuple: A tuple with two elements:
                - Tensor: Output from the student model (`student_model` method).
                - Tensor: Output from the teacher model (computed without gradients).
        """
        
        with torch.no_grad():
            teacher_out = self.teacher_model(x)
        
        student_out = self.student_model(x)

        return (student_out, teacher_out)