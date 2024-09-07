import numpy as np
import torch
import backbones
import torch.nn as nn

from configs.config import setup_config
from cv.datasets import ClassificationDataset
from src.checkpoints import Checkpoint
from torchvision import ops
from utils.bbox_utils import batch_bbox_ious
from utils.global_params import Global

class Stage1(nn.Module):

    def __init__(self,
                 backbone_name,
                 backbone_params="default",
                 anchor_box_scales=[2, 4, 6],
                 anchor_box_aspect_ratios=[0.5, 1, 1.5],
                 anchor_box_boundary_threshold=0.5,
                 anchor_box_positive_threshold=0.7):

        super(Stage1, self).__init__()

        if isinstance(backbone_params, str) and backbone_params == "default": backbone_params = {}
        backbone_model = getattr(backbones, backbone_name)(**backbone_params)
        backbone_model = Checkpoint.load(
            model=backbone_model,
            name=backbone_name
        )
        if backbone_name == "VGG16":
            self.backbone = backbone_model.feature_extractor[:-1]
        else:
            self.backbone = backbone_model.backbone
        self.anchor_box_scales = anchor_box_scales
        self.anchor_bos_aspect_ratios = anchor_box_aspect_ratios
        self.anchor_box_boundary_threshold = anchor_box_boundary_threshold
        self.anchor_box_positive_threshold = anchor_box_positive_threshold

    def _generate_anchor_boxes(self, batch_size, return_projected_boxes=False):
        grid_x, grid_y = np.meshgrid(
            np.arange(self.feature_width),
            np.arange(self.feature_height)
        )

        scales, aspect_ratios = np.meshgrid(self.anchor_box_scales, self.anchor_bos_aspect_ratios)
        widths = aspect_ratios.ravel() * scales.ravel()
        heights = scales.ravel()

        anchor_boxes = []
        for width, height in zip(widths, heights):
            x1 = np.clip(grid_x - width / 2, self.anchor_box_boundary_threshold, self.feature_width - self.anchor_box_boundary_threshold + 0.5)
            y1 = np.clip(grid_y - height / 2, self.anchor_box_boundary_threshold, self.feature_height - self.anchor_box_boundary_threshold + 0.5)
            x2 = np.clip(grid_x + width / 2, self.anchor_box_boundary_threshold, self.feature_width - self.anchor_box_boundary_threshold + 0.5)
            y2 = np.clip(grid_y + height / 2, self.anchor_box_boundary_threshold, self.feature_height - self.anchor_box_boundary_threshold + 0.5)

            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            anchor_boxes.append(boxes.reshape(-1, 4))

        anchor_boxes = np.vstack(anchor_boxes)
        self.anchor_boxes = torch.from_numpy(anchor_boxes).unsqueeze(0).expand(batch_size, -1, -1)

        if return_projected_boxes: # to be removed later
            self.projected_anchor_boxes = np.zeros_like(anchor_boxes)
            self.projected_anchor_boxes[:, 0::2] = anchor_boxes[:, 0::2] * self.width_scale_factor  # Placeholder
            self.projected_anchor_boxes[:, 1::2] = anchor_boxes[:, 1::2] * self.height_scale_factor  # Placeholder

    def _get_features(self, x):

        B, _, _, _ = x.shape

        _, _, self.image_height, self.image_width = x.shape
        x = self.backbone(x)
        _, _, self.feature_height, self.feature_width = x.shape

        self.height_scale_factor = self.image_height // self.feature_height
        self.width_scale_factor = self.image_width // self.feature_width

        if not hasattr(self, "anchor_boxes"):
            self._generate_anchor_boxes(batch_size=B, return_projected_boxes=True) # to be set to False

        return x
    
    @staticmethod
    def _calculate_box_offsets(pos_anc_coords, gt_bbox_mapping):
        pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
        gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

        gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
        anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

        tx_ = (gt_cx - anc_cx)/anc_w
        ty_ = (gt_cy - anc_cy)/anc_h
        tw_ = torch.log(gt_w / anc_w)
        th_ = torch.log(gt_h / anc_h)

        return torch.stack([tx_, ty_, tw_, th_], dim=-1)

    
    def _prepare_anchor_boxes(self, gts):

        B, N, _ = gts.shape

        gts, classes = gts[:, :, :-1], gts[:, :, -1]

        invalid_gts = gts == -1

        gts[:, :, 0::2] = gts[:, :, 0::2] * self.feature_width
        gts[:, :, 1::2] = gts[:, :, 1::2] * self.feature_height

        gts.masked_fill_(invalid_gts, -1)

        iou_matrix = batch_bbox_ious(self.anchor_boxes, gts)

        max_iou_per_gt_box, _ = iou_matrix.max(dim=1, keepdim=True)
        positive_anchor_mask = torch.logical_and(iou_matrix == max_iou_per_gt_box, max_iou_per_gt_box > 0.0)
        positive_anchor_mask = torch.logical_or(positive_anchor_mask, iou_matrix > self.anchor_box_positive_threshold)
        
        positive_anchor_indices_sep = torch.where(positive_anchor_mask)[0]

        positive_anchor_mask = positive_anchor_mask.flatten(start_dim=0, end_dim=1)
        positive_anchor_indices = torch.where(positive_anchor_mask)[0]

        max_iou_per_anchor, max_iou_per_anchor_indices = iou_matrix.max(dim=-1)
        max_iou_per_anchor = max_iou_per_anchor.flatten(start_dim=0, end_dim=1)

        gt_conf_scores = max_iou_per_anchor[positive_anchor_indices]

        gt_classes_expand = classes.view(B, 1, N).expand(B, self.anchor_boxes.size(1), N)
        gt_classes = torch.gather(gt_classes_expand, -1, max_iou_per_anchor_indices.unsqueeze(-1)).squeeze(-1)
        gt_classes = gt_classes.flatten(start_dim=0, end_dim=1)
        gt_classes_positive = gt_classes[positive_anchor_indices]

        gt_bboxes_expand = gts.view(B, 1, N, 4).expand(B, self.anchor_boxes.size(1), N, 4)
        gt_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anchor_indices.reshape(B, self.anchor_boxes.size(1), 1, 1).repeat(1, 1, 1, 4))

        gt_bboxes = gt_bboxes.flatten(start_dim=0, end_dim=2)
        gt_bboxes_positive = gt_bboxes[positive_anchor_indices]

        anchor_boxes_flat = self.anchor_boxes.flatten(start_dim=0, end_dim=-2)
        positive_anchor_coords = anchor_boxes_flat[positive_anchor_indices]
        
        gts_offsets = self._calculate_box_offsets(positive_anchor_coords, gt_bboxes_positive)

        negative_anchor_mask = (max_iou_per_anchor < self.anchor_box_negative_threshold)
        negative_anchor_indices = torch.where(negative_anchor_mask)[0]
        negative_anchor_indices = negative_anchor_indices[torch.randint(0, negative_anchor_indices.shape[0], (positive_anchor_indices.shape[0],))]
        negative_anchor_coords = anchor_boxes_flat[negative_anchor_indices]

        return positive_anchor_indices, negative_anchor_indices, gt_conf_scores, gts_offsets, gt_classes_positive, \
         positive_anchor_coords, negative_anchor_coords, positive_anchor_indices_sep
        

    def forward(self, x):
        x, gts = x

        images = ClassificationDataset.INVERSE_TRANSFORM(x)

        x = self._get_features(x)
        
        self._prepare_anchor_boxes(gts)

        return x       
