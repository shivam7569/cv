import torch.nn.functional as NF
import torchvision.transforms.functional as F
from utils.training import ConditionalRandomFields
from torchvision.transforms import InterpolationMode

class DeepLabEval:

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

