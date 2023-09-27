import os
import cv2
from torch.utils.data import Dataset

from coco.coco_parser import CocoParser
from global_params import Global

class DetectionDataset(Dataset):

    def __init__(self, phase):

        Global.LOGGER.info(f"Parsing coco detection data for {phase}")

        self.coco_parser = CocoParser(phase=phase)
        if phase == "train":
            self.img_dir = Global.CFG.DATA.TRAIN_IMAGES_DIR
        elif phase == "val":
            self.img_dir = Global.CFG.DATA.VAL_IMAGES_DIR
        else:
            Global.LOGGER.error("Invalid phase")

    def __len__(self):
        return len(self.coco_parser.imgIds)
    
    def __getitem__(self, index):
        img_id = self.coco_parser.imgIds[index]

        img_info = self.coco_parser.getImgInfo(img_id=img_id)
        img_path = os.path.join(self.img_dir, img_info["filename"])

        img = cv2.imread(img_path, -1)
        annotations = self.coco_parser.getDetectionAnnotation(img_id=img_id)

        return img, annotations
