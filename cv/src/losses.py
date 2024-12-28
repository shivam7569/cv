import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from cv.utils import MetaWrapper

class InceptionLoss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Loss function as introduced in Inception paper"

    def __init__(self, ce_loss_params):

        super(InceptionLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss(
            **ce_loss_params
        )

    def forward(self, outputs, labels, phase="eval"):

        if phase == "eval":
            return self.ce(outputs, labels)

        aux_classifier_1_out, aux_classifier_2_out, main_classifier_out = outputs

        aux_classifier_1_loss = self.ce(aux_classifier_1_out, labels)
        aux_classifier_2_loss = self.ce(aux_classifier_2_out, labels)
        main_classifier_loss = self.ce(main_classifier_out, labels)

        loss = main_classifier_loss + 0.3 * (aux_classifier_1_loss + aux_classifier_2_loss)

        return loss


def inception_loss(outputs, labels, primitive_loss_fn):
    aux_classifier_1_out, aux_classifier_2_out, main_classifier_out = outputs
    
    aux_classifier_1_loss = primitive_loss_fn(aux_classifier_1_out, labels)
    aux_classifier_2_loss = primitive_loss_fn(aux_classifier_2_out, labels)
    main_classifier_loss = primitive_loss_fn(main_classifier_out, labels)

    loss = main_classifier_loss + 0.3 * (aux_classifier_1_loss + aux_classifier_2_loss)

    return loss

def inceptionv2_loss(outputs, labels, primitive_loss_fn):
    aux_classifier_out, main_classifier_out = outputs
    
    aux_classifier_loss = primitive_loss_fn(aux_classifier_out, labels)
    main_classifier_loss = primitive_loss_fn(main_classifier_out, labels)

    loss = main_classifier_loss + (0.3 * aux_classifier_loss)

    return loss

class Inceptionv3Loss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Loss function as introduced in Inceptionv3 paper"

    def __init__(self, **params):

        super(Inceptionv3Loss, self).__init__()

        self.aux_loss_weightage = params.pop("aux_loss_weightage", 0.3)
        self.ce = nn.CrossEntropyLoss(
            **params
        )

    def forward(self, outputs, labels, phase="eval"):

        if phase == "eval":
            return self.ce(outputs, labels)

        aux_out, out = outputs

        aux_classifier_loss = self.ce(aux_out, labels)
        main_classifier_loss = self.ce(out, labels)

        loss = main_classifier_loss + self.aux_loss_weightage * aux_classifier_loss

        return loss

class OHEM_Loss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Online Hard Example Mining Loss function"

    def __init__(self, pos_weight=1.0, neg_weight=1.0, top_k_ratio=3):
        super(OHEM_Loss, self).__init__()

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.top_k_ratio = top_k_ratio

    def forward(self, outputs, targets):
        pos_loss = -torch.log(outputs[:, targets])

        neg_loss = -torch.log(1 - outputs)

        num_neg = min(int(self.top_k_ratio * len(targets)), len(outputs) - len(targets))
        neg_loss, _ = torch.topk(neg_loss[:, 1:], num_neg, dim=1)

        loss = (self.pos_weight * pos_loss.sum() + self.neg_weight * neg_loss.sum()) / len(outputs)

        return loss

class DeiT_Distillation_Loss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Distillation Loss function as introduced in DeiT paper"

    def __init__(self, ce_loss_params, kl_loss_params=None, balance_coefficient=0.1, temperature=3.0, distillation_kind="hard"):

        super(DeiT_Distillation_Loss, self).__init__()

        self.ce = nn.CrossEntropyLoss(**ce_loss_params)

        self.distillation_kind = distillation_kind
        if distillation_kind == "soft":

            assert kl_loss_params is not None
            assert balance_coefficient is not None
            assert temperature is not None

            self.kl_divergence = nn.KLDivLoss(**kl_loss_params)
            self.temperature = temperature
            self.balance_coefficient = balance_coefficient

    def soft_distillation(self, student_logits, teacher_logits, target):
        kl_part_loss = self.balance_coefficient * (self.temperature ** 2) * self.kl_divergence(
            F.softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        )

        ce_part_loss = (1 - self.balance_coefficient) * self.ce(student_logits, target)

        loss = kl_part_loss + ce_part_loss

        return loss

    def hard_distillation(self, student_logits, teacher_target, target):

        loss = (self.ce(student_logits, target) + self.ce(student_logits, teacher_target)) / 2

        return loss
    
    def forward(self, preds, targets):

        """
            preds[0]: student_logits
            preds[1]: teacher_logits
            target: true labels
        """

        if self.distillation_kind == "hard":
            teacher_target = F.softmax(preds[1], dim=1).argmax(dim=1)
            return self.hard_distillation(
                student_logits=preds[0],
                teacher_target=teacher_target,
                target=targets
            )
        if self.distillation_kind == "soft":
            return self.soft_distillation(
                student_logits=preds[0],
                teacher_logits=preds[1],
                target=targets
            )


class DiceLoss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Dice Loss function for segmentation use cases"

    def __init__(
            self,
            weight,
            num_classes,
            reduction="mean",
            log_loss: bool=False,
            log_cosh: bool=False,
            normalize: bool=False,
            smooth: float=0.0,
            ignore_index: int=-1,
            classes: torch.Tensor=None,
            eps: float=1e-7
    ):
        
        super(DiceLoss, self).__init__()

        self.num_classes = num_classes
        self.log_loss = log_loss
        self.log_cosh = log_cosh
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.classes = classes
        self.reduction = reduction
        self.weights = weight
        self.normalize = normalize
        self.eps = eps

        if log_cosh: assert log_loss == False
        if log_loss: assert log_cosh == False

    def forward(self, preds: torch.Tensor, gts: torch.Tensor):

        """
        preds: shape === (B, C, H, W)
        gts: shape === (B, 1, H, W)
        """

        if not gts.shape[0]: return 0.0

        gts = gts.squeeze(1)

        B, H, W = gts.size()

        ignore_index_mask = gts == self.ignore_index
        preds = F.log_softmax(preds, dim=1).exp().permute(0, 2, 3, 1)

        if ignore_index_mask.sum():
            gts[ignore_index_mask] = self.num_classes
            gts = F.one_hot(gts, self.num_classes + 1)
            gts = gts[...,:-1]
        else:
            gts = F.one_hot(gts, self.num_classes)
        
        if self.classes is not None:
            indices = self.classes[None, None, None, :].expand(B, H, W, -1)
            gts = gts.gather(dim=-1, index=indices)
            preds = preds.gather(dim=-1, index=indices)

        gts = gts.permute(0, 3, 1, 2)
        preds = preds.permute(0, 3, 1, 2)

        intersection = torch.sum(gts * preds, dim=(0, 2, 3))
        cardinality = torch.sum(gts + preds, dim=(0, 2, 3))

        dice_score = torch.clamp_min((2 * intersection + self.smooth) / (cardinality + self.smooth), min=self.eps)

        if self.log_loss:
            loss = -dice_score.log()
        else:
            loss = 1 - dice_score

        if self.weights is not None:
            loss *= self.weights

        if self.log_cosh:
            loss = loss.cosh().log()

        if self.normalize:
            loss /= loss.sum()

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError("Invalid reduction method")

        return loss

class TverskyLoss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Tversky Loss function for segmentation use cases"

    def __init__(
            self,
            weight,
            num_classes,
            alpha=0.5,
            beta=0.5,
            gamma=1,
            reduction="mean",
            log_loss: bool=False,
            normalize=False,
            smooth: float=0.0,
            ignore_index: int=-1,
            classes: torch.Tensor=None,
            eps: float=1e-7
    ):
        
        super(TverskyLoss, self).__init__()

        self.num_classes = num_classes
        self.log_loss = log_loss
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.classes = classes
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, preds: torch.Tensor, gts: torch.Tensor):

        """
        preds: shape === (B, C, H, W)
        gts: shape === (B, 1, H, W)
        """

        if not gts.shape[0]: return 0.0

        gts = gts.squeeze(1)

        B, H, W = gts.size()

        ignore_index_mask = gts == self.ignore_index
        preds = F.log_softmax(preds, dim=1).exp().permute(0, 2, 3, 1)

        if ignore_index_mask.sum():
            gts[ignore_index_mask] = self.num_classes
            gts = F.one_hot(gts, self.num_classes + 1)
            gts = gts[...,:-1]
        else:
            gts = F.one_hot(gts, self.num_classes)
        
        if self.classes is not None:
            indices = self.classes[None, None, None, :].expand(B, H, W, -1)
            gts = gts.gather(dim=-1, index=indices)
            preds = preds.gather(dim=-1, index=indices)

        gts = gts.permute(0, 3, 1, 2)
        preds = preds.permute(0, 3, 1, 2)

        tp = torch.sum(gts * preds, dim=(0, 2, 3))
        fp = torch.sum(preds * (1 - gts), dim=(0, 2, 3))
        fn = torch.sum(gts * (1 - preds), dim=(0, 2, 3))

        tversky_index = torch.clamp_min((tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth), min=self.eps)

        if self.log_loss:
            loss = -self.gamma * tversky_index.log()
        else:
            loss = torch.pow(1 - tversky_index, self.gamma)

        if self.weights is not None:
            loss *= self.weights

        if self.normalize:
            loss /= loss.sum()

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError("Invalid reduction method")

        return loss
    
class FocalLoss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Focal Loss function for segmentation use cases"

    def __init__(
            self,
            weight=None,
            gamma=2,
            ignore_index=-1,
            reduction="none", 
            focal_reduction="mean",
            label_smoothing=0.0,
            normalize=False
        ):

        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.normalize = normalize
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
            )
        self.focal_reduction = focal_reduction

    @staticmethod
    def flatten(pred, target, ignore_index):

        num_class = pred.size(1)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        
        input_flatten = pred.view(-1, num_class)
        target_flatten = target.view(-1)

        mask = (target_flatten != ignore_index)
        input_flatten = input_flatten[mask]
        target_flatten = target_flatten[mask]
        
        return input_flatten, target_flatten
      
    def forward(self, pred, target):
        pred, target = self.flatten(pred, target, self.ignore_index)
        input_prob = torch.gather(F.softmax(pred, dim=1), 1, target.unsqueeze(1))
        cross_entropy = self.ce(pred, target)
        losses = (1 - input_prob).pow_(self.gamma).squeeze_(1) * cross_entropy

        if self.normalize:
            losses /= losses.sum()
        
        if self.focal_reduction == "mean":
            loss = losses.mean()
        if self.focal_reduction == "sum":
            loss = losses.sum()

        return loss
    
class ComboLoss(nn.Module, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Loss function to combine multiple losses for segmentation use cases"

    def __init__(self, weight=None, _lambda=0.5, dynamic_weighting=False, alpha=0.99, **kwargs):

        super(ComboLoss, self).__init__()

        self._lambda = _lambda
        self.alpha = alpha
    
        self.dynamic_weighting = dynamic_weighting
        if self.dynamic_weighting:
            self.running_avg1 = 1.0
            self.running_avg2 = 1.0

        self.focal_loss = FocalLoss(weight=weight, **kwargs["focal_params"])
        self.dice_loss = DiceLoss(weight=None, **kwargs["dice_params"])

    def forward(self, preds, gts):
        focal_loss = self.focal_loss(preds, gts)
        dice_loss = self.dice_loss(preds, gts)

        if not self.dynamic_weighting:
            loss = self._lambda * focal_loss + (1 - self._lambda) * dice_loss
        else:
            self.running_avg1 = self.alpha * self.running_avg1 + (1 - self.alpha) * focal_loss.item()
            self.running_avg2 = self.alpha * self.running_avg2 + (1 - self.alpha) * dice_loss.item()

            weight1 = 1.0 / (self.running_avg1 + 1e-8)
            weight2 = 1.0 / (self.running_avg2 + 1e-8)
            weight_sum = weight1 + weight2
            weight1 /= weight_sum
            weight2 /= weight_sum

            loss = weight1 * focal_loss + weight2 * dice_loss

        return loss

class DeepLabv1Loss(nn.Module, metaclass=MetaWrapper):

    def __init__(self, weight=None, **kwargs):
        super(DeepLabv1Loss, self).__init__()
        self.loss_name = kwargs.pop("name", "dice")
        self._lambda = kwargs.pop("_lambda", 0.5)
        if self.loss_name == "ce":
            self.loss = nn.CrossEntropyLoss(weight=weight, **kwargs)
        if self.loss_name == "dice":
            self.loss = DiceLoss(weight=weight, **kwargs)
        if self.loss_name == "tversky":
            self.loss = TverskyLoss(weight=weight, **kwargs)
        if self.loss_name == "tversky_ce":
            tversky_weights = kwargs.pop("tversky_weights", False)
            self.first_loss = nn.CrossEntropyLoss(weight=weight, **kwargs["ce_params"])
            self.second_loss = TverskyLoss(weight=weight if tversky_weights else None, **kwargs["tversky_params"])
        if self.loss_name == "dice_ce":
            dice_weights = kwargs.pop("dice_weights", False)
            self.first_loss = nn.CrossEntropyLoss(weight=weight, **kwargs["ce_params"])
            self.second_loss = DiceLoss(weight=weight if dice_weights else None, **kwargs["dice_params"])
        if self.loss_name == "combo":
            self.loss = ComboLoss(weight=weight, _lambda=self._lambda, **kwargs)

    def forward(self, preds: torch.Tensor, gts: torch.Tensor):
        gts = TF.resize(img=gts, size=preds.shape[-2:], interpolation=TF.InterpolationMode.NEAREST, antialias=True)
        
        if self.loss_name not in ["tversky_ce", "dice_ce"]:
            loss = self.loss(preds, gts)
        else:
            first_lose = self.first_loss(preds, gts)
            second_loss = self.second_loss(preds, gts)
            loss = (1 - self._lambda) * first_lose + self._lambda * second_loss

        return loss

class NT_XentLoss(nn.Module):

    def __init__(self, temperature: float):

        super(NT_XentLoss, self).__init__()

        self.temperature = temperature

    @staticmethod
    def get_pair_indices(batch_size):

        arrange_indices, pairs = [], []

        for i in range(batch_size):
            arrange_indices.extend([i, i+batch_size])

        for i in range(0, 2*batch_size, 2):
            pairs.append((range(2*batch_size)[i], range(2*batch_size)[i+1]))
            pairs.append((range(2*batch_size)[i+1], range(2*batch_size)[i]))

        return (arrange_indices, torch.tensor(pairs))

    def forward(self, embeddings):

        B, _ = embeddings.shape
        B = B // 2

        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        arrange_indices, pair_indices = self.get_pair_indices(B)

        embeddings = embeddings[arrange_indices]

        similarity = embeddings @ embeddings.t()
        similarity = (similarity / self.temperature).exp()

        numerator = similarity[pair_indices[:, 0], pair_indices[:, 1]]
        denominator = similarity.sum(dim=1) - similarity.diag()

        loss = (-torch.log(numerator / denominator)).sum() / (2 * B)

        return loss
