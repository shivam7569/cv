import cv2
import numpy as np
from tqdm import tqdm
import torch

from src.gpu_devices import GPU_Support

from src.metrics import DetectionMetrics
from utils.pytorch_utils import numpy2tensor

class Eval:
    
    def __init__(
        self,
        model,
        algorithm,
        data_loader,
        loss_function,
        tb_writer=None,
    ):
        
        self.model = model
        self.algorithm = algorithm
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.tb_writer = tb_writer

        self.metrics = DetectionMetrics()

    def start(self, epoch=None):
        self.model.eval()

        if self.tb_writer is not None: self.tb_writer.setWriter("val")

        data_iterator = tqdm(self.data_loader, desc="Evaluating", unit="batch")
        loss = 0

        for batch in data_iterator:
            model_inputs = []

            if self.algorithm == "FastRCNN":
                img_batch, rois = batch
                coordinates = rois[:, :, :4]
                target = numpy2tensor(rois[:, :, 4:])

                model_inputs.append(img_batch)
                model_inputs.append(coordinates)

            output = self.model(*model_inputs)

            self._process_outputs(model_inputs, output)

    def _process_outputs(self, model_ins, model_outs):

        # this needs to be fixed
        if self.algorithm == "FastRCNN":
            images = model_ins[0]
            rois = model_ins[1]
            
            feature_h, feature_w = images.shape[2:]
            rois[:, :, [0, 2]] = rois[:, :, [0, 2]] * feature_w
            rois[:, :, [1, 3]] = rois[:, :, [1, 3]] * feature_h

            rois = rois.astype(int)
            rois = np.reshape(rois, (-1, 4)) # in [x1, y1, x2, y2] format

            classification_out = model_outs[0]
            classification_scores = torch.softmax(classification_out, dim=1)
            class_scores, class_ids = torch.max(classification_scores, dim=1)
            non_background_indices = (class_ids != 0).nonzero().squeeze()

            if non_background_indices.numel() == 0: return []

            non_background_classes = class_ids[non_background_indices]
            non_background_scores = class_scores[non_background_indices]
            non_background_classes -= 1

            rois = rois[non_background_indices]

            regression_out = model_outs[1][non_background_indices]
            regression_out = regression_out.view(regression_out.shape[0], -1, 4)

            box_offsets = torch.gather(regression_out, 1, non_background_classes.view(-1, 1, 1).expand(-1, 1, 4)).squeeze().detach().cpu().numpy() # [tx, ty, tw, th]

            centered_rois = np.ones_like(rois)
            centered_rois[:, 0] = np.sum(rois[:, [0, 2]], axis=1) // 2
            centered_rois[:, 1] = np.sum(rois[:, [1, 3]], axis=1) // 2
            centered_rois[:, 2] = rois[:, 2] - rois[:, 0]
            centered_rois[:, 3] = rois[:, 3] - rois[:, 1]

            centered_predictions = np.ones_like(rois)
            centered_predictions[:, 0] = centered_rois[:, 2] * box_offsets[:, 0] + centered_rois[:, 0]
            centered_predictions[:, 1] = centered_rois[:, 3] * box_offsets[:, 1] + centered_rois[:, 1]
            centered_predictions[:, 2] = centered_rois[:, 2] * np.exp(box_offsets[:, 2])
            centered_predictions[:, 3] = centered_rois[:, 3] * np.exp(box_offsets[:, 3])

            predictions = np.ones_like(centered_predictions)
            predictions[:, 0] = centered_predictions[:, 0] - (centered_predictions[:, 2] // 2)
            predictions[:, 1] = centered_predictions[:, 1] - (centered_predictions[:, 3] // 2)
            predictions[:, 2] = centered_predictions[:, 0] + (centered_predictions[:, 2] // 2)
            predictions[:, 3] = centered_predictions[:, 1] + (centered_predictions[:, 3] // 2)

            scores = non_background_scores.view(-1, 1).detach().cpu().numpy()
            c_ids = non_background_classes.view(-1, 1).detach().cpu().numpy()
            predictions = np.concatenate([predictions, scores, c_ids], axis=1)
            predictions = self.metrics.nms(predictions)
            img = cv2.imread("workshop/Example.png")
            for r in predictions[:, :4].astype(int):
                x1, y1, x2, y2 = r
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

            cv2.imwrite("workshop/Ex.png", img)
            a = 1