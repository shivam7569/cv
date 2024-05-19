import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionLoss(nn.Module):

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

class OHEM_Loss(nn.Module):

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

class DeiT_Distillation_Loss(nn.Module):

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


class DiceLoss(nn.Module):

    def __init__(
            self,
            num_classes,
            reduction="mean",
            log_loss: bool=False,
            smooth: float=0.0,
            ignore_index: int=-1,
            classes: torch.Tensor=None,
            eps: float=1e-7
    ):
        
        super(DiceLoss, self).__init__()

        self.num_classes = num_classes
        self.log_loss = log_loss
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.classes = classes
        self.reduction = reduction
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

        intersection = torch.sum(gts * preds, dim=(0, 2, 3))
        cardinality = torch.sum(gts + preds, dim=(0, 2, 3))

        dice_score = torch.clamp_min((2 * intersection + self.smooth) / (cardinality + self.smooth), min=self.eps)

        if self.log_loss:
            loss = -dice_score.log()
        else:
            loss = 1 - dice_score

        existence_class_mask = torch.sum(gts, dim=(0, 2, 3)) > 0
        loss *= existence_class_mask

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError("Invalid reduction method")

        return loss
