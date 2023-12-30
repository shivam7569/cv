import torch
import torch.nn as nn

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
