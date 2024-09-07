import torch
import torch.nn as nn

from cv import backbones
from cv.utils import Global
from cv.backbones import ViT
from cv.src.checkpoints import Checkpoint
from cv.utils import MetaWrapper

class DeiT(ViT, metaclass=MetaWrapper):

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

        Global.LOGGER.info(f"Using {self.teacher_model_name} as teacher model")

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
        self.teacher_model = Checkpoint.load(
            model=self.teacher_model, name=self.teacher_model_name
        )
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_model.eval()

    def student_model(self, x):

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
        with torch.no_grad():
            teacher_out = self.teacher_model(x)
        
        student_out = self.student_model(x)

        return (student_out, teacher_out)