import torch
from torch import Tensor

def batch_bbox_ious(boxes_1: Tensor, boxes_2: Tensor):

    """
    boxes_1: (B, N, 4) and boxes_2: (B, M, 4)
    Both are in the format of (x1, y1, x2, y2)
    """
    
    boxes_1_wh = boxes_1[:, :, [2, 3]] - boxes_1[:, :, [0, 1]]
    boxes_2_wh = boxes_2[:, :, [2, 3]] - boxes_2[:, :, [0, 1]]

    boxes_1_areas = boxes_1_wh[:, :, 0] * boxes_1_wh[:, :, 1]
    boxes_2_areas = boxes_2_wh[:, :, 0] * boxes_2_wh[:, :, 1]

    intersection_x1_y1 = torch.max(boxes_1[:, :, None, [0, 1]], boxes_2[:, None, :, [0, 1]])
    intersection_x2_y2 = torch.min(boxes_1[:, :, None, [2, 3]], boxes_2[:, None, :, [2, 3]])

    intersection_wh = (intersection_x2_y2 - intersection_x1_y1).clamp(min=0.0)
    intersection = intersection_wh[:, :, :, 0] * intersection_wh[:, :, :, 1]

    union = boxes_1_areas[:, :, None] + boxes_2_areas[:, None, :] - intersection

    ious = intersection / union

    return ious

def bbox_ious(boxes_1: Tensor, boxes_2: Tensor):

    """
    boxes_1: (N, 4) and boxes_2: (M, 4)
    Both are in the format of (x1, y1, x2, y2)
    """
    
    boxes_1_wh = boxes_1[:, [2, 3]] - boxes_1[:, [0, 1]]
    boxes_2_wh = boxes_2[:, [2, 3]] - boxes_2[:, [0, 1]]

    boxes_1_areas = boxes_1_wh[:, 0] * boxes_1_wh[:, 1]
    boxes_2_areas = boxes_2_wh[:, 0] * boxes_2_wh[:, 1]

    intersection_x1_y1 = torch.max(boxes_1[:, None, [0, 1]], boxes_2[:, [0, 1]])
    intersection_x2_y2 = torch.min(boxes_1[:, None, [2, 3]], boxes_2[:, [2, 3]])

    intersection_wh = (intersection_x2_y2 - intersection_x1_y1).clamp(min=0.0)
    intersection = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    union = boxes_1_areas[:, None] + boxes_2_areas - intersection

    ious = intersection / union

    return ious
