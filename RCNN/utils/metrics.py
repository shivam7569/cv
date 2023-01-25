import numpy as np

from RCNN.utils.globalParams import Global


def iou(pred_box, target_boxes):
    "Compute the IoU of candidate proposal and labeled bounding boxes"

    if len(target_boxes.shape) == 1:
        target_boxes = target_boxes[np.newaxis, :]
    
    xA = np.maximum(pred_box[0], target_boxes[:, 0])
    yA = np.maximum(pred_box[1], target_boxes[:, 1])
    xB = np.minimum(pred_box[2], target_boxes[:, 2])
    yB = np.minimum(pred_box[3], target_boxes[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)

    return scores

def non_max_suppression(rect_list, score_list):

    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = Global.NON_MAX_SUPPRESSION_IOU_THRESHOLD
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        iou_scores = iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores

def softmax(values):

    e = np.e
    exp_val = [e**i for i in values]
    sum_exp_val = sum(exp_val)

    softmax_values = [i / sum_exp_val for i in exp_val]

    return softmax_values