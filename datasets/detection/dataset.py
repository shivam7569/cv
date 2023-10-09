import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from datasets.coco_parser import CocoParser
from utils.global_params import Global

class DetectionDataset(Dataset):

    def __init__(self, phase):

        Global.LOGGER.info(f"Parsing coco detection data for {phase}")

        self.coco_parser = CocoParser(phase=phase)
        if phase == "train":
            self.img_dir = Global.CFG.DATA.COCO_TRAIN_IMAGES_DIR
        elif phase == "val":
            self.img_dir = Global.CFG.DATA.COCO_VAL_IMAGES_DIR
        else:
            Global.LOGGER.error("Invalid phase")

    def __len__(self):
        return len(self.coco_parser.imgIds)
    
    def __getitem__(self, index):

        # bbox are in: [x1, y1, x2, y2]

        img_id = self.coco_parser.imgIds[index]

        img_info = self.coco_parser.getImgInfo(img_id=img_id)
        img_path = os.path.join(self.img_dir, img_info["filename"])

        img = cv2.imread(img_path, -1)
        annots = self.coco_parser.getDetectionAnnotation(img_id=img_id)

        annotations = np.zeros((len(annots["bboxes"]), 5), dtype=np.float32)

        img_height, img_width = img_info["height"], img_info["width"]

        bboxes = np.array(annots["bboxes"])
        np.clip(bboxes[:, 0], 0, img_width, out=bboxes[:, 0])
        np.clip(bboxes[:, 1], 0, img_height, out=bboxes[:, 1])
        np.clip(bboxes[:, 2], 0, img_width, out=bboxes[:, 2])
        np.clip(bboxes[:, 3], 0, img_height, out=bboxes[:, 3])

        for i, box in enumerate(bboxes.tolist()):
            annotations[i, 0] = box[0]
            annotations[i, 1] = box[1]
            annotations[i, 2] = box[2]
            annotations[i, 3] = box[3]
            annotations[i, 4] = annots["categories"][i]

        return img, annotations
