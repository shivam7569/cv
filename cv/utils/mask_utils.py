import cv2
import json
import numpy as np

from cv.utils import Global

def getColorMask(class_mask):

    with open(Global.CFG.DATA.COCO_ID_TO_COLOR, "r") as f:
        coco_colors = json.load(f)

    height, width = class_mask.shape
    color_mask = np.zeros(shape=(height, width), dtype=np.uint8).reshape(height, width, 1).repeat(3, axis=2)

    unique_classes = np.unique(class_mask)

    for cls_id in unique_classes:
        try:
            color_mask[class_mask == cls_id] = coco_colors[str(cls_id)]
        except:
            color_mask[class_mask == cls_id] = [0, 0, 0]

    return color_mask

def getOverlay(img, mask):
    overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)

    return overlay