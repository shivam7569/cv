import torch.nn.functional as NF

from cv.utils import MetaWrapper
from cv.utils.training import ConditionalRandomFields

class DeepLabEval(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Utility class to support CRF post processing for the evaluation of DeepLab segmentation model"

    def __init__(self, **kwargs):
        self.crf = ConditionalRandomFields(
            interpolation_params=kwargs.pop("interpolation_params"),
            crf_params=kwargs.pop("crf_params")
        )

    def __call__(self, images, masks, outputs):

        self.outputs, self.pro_outputs = self.crf.mp_process(img_batch=images, output_batch=outputs)
        setattr(self, "masks", masks)

    def loss(self, loss_fn):
        loss_val = loss_fn(self.outputs, self.masks.squeeze(1)).item()

        return loss_val
    
    def predicted_mask(self):
        return NF.log_softmax(self.pro_outputs, dim=1).exp().argmax(dim=1)
    
    def metric_update(self, metric_obj):
        metric_obj.update(self.masks.squeeze(1), self.pro_outputs)

