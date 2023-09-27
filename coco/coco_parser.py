import json
import numpy as np
from configs.config import get_cfg
from global_params import Global

from pycocotools.coco import COCO

class CocoParser:

    def __init__(self, phase):

        self.cfg = get_cfg()
        if phase == "train":
            self.img_dir = self.cfg.DATA.TRAIN_IMAGES_DIR
            self.annot_path = self.cfg.DATA.TRAIN_ANNOTATIONS
        elif phase == "val":
            self.img_dir = self.cfg.DATA.VAL_IMAGES_DIR
            self.annot_path = self.cfg.DATA.VAL_ANNOTATIONS
        else:
            Global.LOGGER.error(f"Incorrect phase for parsing")

        self.coco = COCO(self.annot_path)
        self.imgIds = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()

        with open(self.annot_path, "r") as f:
            coco_dict = json.load(f)

        self.cat_id_to_name = {}
        for i in coco_dict["categories"]:
            self.cat_id_to_name[i["id"]] = i["name"]
    
    def getDetectionAnnotation(self, img_id):
        assert img_id in self.imgIds, Global.LOGGER.error(f"Image id {img_id} is invalid")
        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)

        annotations = {"bboxes": [], "categories": []}
        for i in anns:
            x1, y1, w, h = [round(j) for j in i["bbox"]]
            annotations["bboxes"].append([x1, y1, x1+w, y1+h])
            annotations["categories"].append(i["category_id"])

        assert len(annotations["bboxes"]) == len(annotations["categories"])

        return annotations
    
    def getSegmentationMask(self, img_id):
        assert img_id in self.imgIds, Global.LOGGER.error(f"Image id {img_id} is invalid")

        image_info = self.coco.loadImgs(img_id)[0]
        height, width = image_info['height'], image_info['width']
        class_mask = np.zeros((height, width), dtype=np.uint8)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds))
        
        for annotation in annotations:
            category_id = annotation['category_id']
            mask = self.coco.annToMask(annotation)
            class_mask[mask > 0] = category_id

        return class_mask
    
    def getImgInfo(self, img_id):
        img_info = {
            "filename": self.coco.loadImgs(img_id)[0]["file_name"],
            "height": self.coco.loadImgs(img_id)[0]["height"],
            "width": self.coco.loadImgs(img_id)[0]["width"]
        }

        return img_info
