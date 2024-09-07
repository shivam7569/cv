import json
import random
import numpy as np
from pycocotools.coco import COCO

from cv.utils import Global
from cv.configs.config import get_cfg
from cv.utils import MetaWrapper

class CocoParser(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to parse and interpret Coco Dataset"

    cfg = get_cfg()

    def __init__(self, phase):

        self.cfg = get_cfg()
        if phase == "train":
            self.img_dir = self.cfg.DATA.COCO_TRAIN_IMAGES_DIR
            self.annot_path = self.cfg.DATA.COCO_TRAIN_ANNOTATIONS
        elif phase == "val":
            self.img_dir = self.cfg.DATA.COCO_VAL_IMAGES_DIR
            self.annot_path = self.cfg.DATA.COCO_VAL_ANNOTATIONS
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

        random.shuffle(self.imgIds)
    
    def getDetectionAnnotation(self, img_id):
        assert img_id in self.imgIds, Global.LOGGER.error(f"Image id {img_id} is invalid")

        coco_id_to_correct_id = CocoParser.getCocoIdToCorrectId()

        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)

        annotations = {"bboxes": [], "categories": []}
        for i in anns:
            x1, y1, w, h = [round(j) for j in i["bbox"]]
            annotations["bboxes"].append([x1, y1, x1+w, y1+h])
            annotations["categories"].append(coco_id_to_correct_id[str(i["category_id"])])

        assert len(annotations["bboxes"]) == len(annotations["categories"])

        return annotations
    
    def getSegmentationMask(self, img_id):
        assert img_id in self.imgIds, Global.LOGGER.error(f"Image id {img_id} is invalid")

        coco_id_to_correct_id = CocoParser.getCocoIdToCorrectId()

        image_info = self.coco.loadImgs(img_id)[0]
        height, width = image_info['height'], image_info['width']
        class_mask = np.zeros((height, width), dtype=np.uint8)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds))
        
        for annotation in annotations:
            category_id = annotation['category_id']
            mask = self.coco.annToMask(annotation)
            class_mask[mask > 0] = coco_id_to_correct_id[str(category_id)]

        return class_mask
    
    def getImgInfo(self, img_id):
        img_info = {
            "filename": self.coco.loadImgs(img_id)[0]["file_name"],
            "height": self.coco.loadImgs(img_id)[0]["height"],
            "width": self.coco.loadImgs(img_id)[0]["width"]
        }

        return img_info
    
    def generateCorrectIds(self):
        correct_ids = {}
        for idx, i in enumerate(self.catIds):
            correct_ids[i] = idx
        
        with open(self.cfg.DATA.COCO_ID_TO_CORRECT_ID, "w") as f:
            json.dump(correct_ids, f)

    @classmethod
    def getCocoIdToCorrectId(cls):
        with open(cls.cfg.DATA.COCO_ID_TO_CORRECT_ID, "r") as f:
            coco_id_to_correct_id = json.load(f)

        return coco_id_to_correct_id

    @classmethod
    def getIdVsName(cls):
        with open(cls.cfg.DATA.COCO_ID_TO_NAME, "r") as f:
            id_vs_name = json.load(f)

        return id_vs_name
    
    @classmethod
    def getExcludeIDs(cls):
        with open(cls.cfg.DATA.COCO_EXCLUDE_IDS, "r") as f:
            exclude_ids = json.load(f)

        return exclude_ids
