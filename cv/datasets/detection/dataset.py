import os
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from cv.utils import Global
from cv.src.custom_transforms import *
from cv.src import pipeline_functions as PF
from cv.datasets.coco_parser import CocoParser
from cv.utils import MetaWrapper

class DetectionDataset(Dataset, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Coco Dataset Class for training of detectors"

    def __init__(self, phase, transforms, log=True):

        if log: Global.LOGGER.info(f"Parsing coco detection data for {phase}")

        self.phase = phase

        self.coco_parser = CocoParser(phase=phase)
        if phase == "train":
            self.img_dir = Global.CFG.DATA.COCO_TRAIN_IMAGES_DIR
        elif phase == "val":
            self.img_dir = Global.CFG.DATA.COCO_VAL_IMAGES_DIR
        else:
            Global.LOGGER.error("Invalid phase")

        if transforms is not None:
            self.transforms = T.Compose(self.parseTransforms(transforms))

        self.coco_parser.imgIds = list(filter(lambda x: len(self.coco_parser.getDetectionAnnotation(img_id=x)["bboxes"]) > 0, self.coco_parser.imgIds))

    def visualizeBBox(self, img, bboxes, filename="workshop/Det_Image.png"):
        for bb in bboxes.astype(int):
            x1, y1, x2, y2, cat = bb
            class_name = self.coco_parser.getIdVsName()[str(int(cat))]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            img = cv2.putText(img, text=class_name, org=(int(x1), int(y1-10)), fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=0.75, color=(255, 255, 255), thickness=1)

        cv2.imwrite(filename, img)

    def executePipeline(self, img_path):
        img_pipeline = Global.CFG.PIPELINES.TRAIN if self.phase == "train" else Global.CFG.PIPELINES.VAL
        if img_pipeline is not None:
            for function in img_pipeline:
                if function["func"] == "readImage":
                    img = getattr(PF, function["func"])(img_path, **function["params"])
                else:
                    img = getattr(PF, function["func"])(img, **function["params"])
        else:
            Global.LOGGER.error("Cannot proceed without data pipeline")

        return img

    def __len__(self):
        return len(self.coco_parser.imgIds)
    
    @classmethod
    def parseTransforms(self, transforms):
        transforms_list = []

        for transform_entry in transforms:
            transform_name = transform_entry["name"]
            transform_params = transform_entry["params"]

            if transform_name == "RandomApply":
                transform_class = getattr(T, transform_name)
                rand_transforms = transform_params["transforms"]
                rand_transforms_list = []

                for random_transform in rand_transforms:
                    rand_transform_name = random_transform["name"]
                    rand_transform_param = random_transform["params"]
                    try:
                        rand_transform_class = getattr(T, rand_transform_name)
                        if rand_transform_param is not None:
                            rand_transform = rand_transform_class(**rand_transform_param)
                        else:
                            rand_transform = rand_transform_class()
                        rand_transforms_list.append(rand_transform)
                    except:
                        if rand_transform_name in globals():
                            rand_transform_instnce = globals()[rand_transform_name](**rand_transform_param)
                            rand_transforms_list.append(rand_transform_instnce)
                
                transforms_list.append(transform_class(transforms=rand_transforms_list, p=transform_params["p"]))
                
            else:
                try:
                    transform_class = getattr(T, transform_name)
                    if transform_params is not None:
                        transform = transform_class(**transform_params)
                    else:
                        transform = transform_class()

                    transforms_list.append(transform)
                except:
                    if transform_name in globals():
                        if transform_name == "RepeatedAugmentation":
                            transform_instnce = globals()[transform_name](
                                transformations=DetectionDataset.parseTransforms(transform_params["transformations"]),
                                repeats=transform_params["repeats"],
                                p=transform_params["p"]
                            )
                        else:
                            transform_instnce = globals()[transform_name](**transform_params)
                        transforms_list.append(transform_instnce)

        return transforms_list

    def collate_fn(self, batch):
        batch = [self.transforms(b) for b in batch]

        images = []
        annotations = []
        max_bounding_boxes = max([i[1].shape[0] for i in batch])

        for batch_entry in batch:
            image = batch_entry[0]
            gts = torch.from_numpy(batch_entry[1])
            gts = F.pad(gts, pad=[0, 0, 0, max_bounding_boxes - gts.shape[0]], value=-1)

            images.append(image)
            annotations.append(gts)

        images = torch.stack(images, dim=0)
        annotations = torch.stack(annotations, dim=0)

        return (images, annotations)
    
    def __getitem__(self, index):

        # bbox are in: [x1, y1, x2, y2]

        img_id = self.coco_parser.imgIds[index]

        img_info = self.coco_parser.getImgInfo(img_id=img_id)
        img_path = os.path.join(self.img_dir, img_info["filename"])

        img = self.executePipeline(img_path)
        annots = self.coco_parser.getDetectionAnnotation(img_id=img_id)

        annotations = np.zeros((len(annots["bboxes"]), 5), dtype=np.float32)

        img_height, img_width = img_info["height"], img_info["width"]

        bboxes = np.array(annots["bboxes"])
        np.clip(bboxes[:, 0], 0, img_width, out=bboxes[:, 0])
        np.clip(bboxes[:, 1], 0, img_height, out=bboxes[:, 1])
        np.clip(bboxes[:, 2], 0, img_width, out=bboxes[:, 2])
        np.clip(bboxes[:, 3], 0, img_height, out=bboxes[:, 3])

        for i, box in enumerate(bboxes.tolist()):
            annotations[i, 0] = box[0] / img_width
            annotations[i, 1] = box[1] / img_height
            annotations[i, 2] = box[2] / img_width
            annotations[i, 3] = box[3] / img_height
            annotations[i, 4] = annots["categories"][i]

        return img, annotations
