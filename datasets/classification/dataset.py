import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from global_params import Global
from utils.file_utils import read_txt

class ClassificationDataset(Dataset):

    def __init__(self, phase, transforms=None, debug=None, log=True):

        if log: Global.LOGGER.info(f"Parsing imagenet classification data for {phase}")

        self.phase = phase
        if phase == "train":
            self.phase_text = Global.CFG.DATA.IMAGENET_TRAIN_TXT
        elif phase == "val":
            self.phase_text = Global.CFG.DATA.IMAGENET_VAL_TXT
        else:
            Global.LOGGER.error("Invalid phase")

        self.img_and_class = read_txt(self.phase_text)

        if debug is not None:
            self.img_and_class = self.img_and_class[:debug]

        self.transforms = T.Compose(self.parseTransforms(transforms))

    def parseTransforms(self, transforms):
        transforms_list = []

        for transform_entry in transforms:
            transform_name = transform_entry["name"]
            transform_params = transform_entry["params"]

            transform_class = getattr(T, transform_name)
            if transform_params is not None:
                transform = transform_class(**transform_params)
            else:
                transform = transform_class()

            transforms_list.append(transform)

        return transforms_list

    def __len__(self):
        return len(self.img_and_class)
    
    def __getitem__(self, index):
        img_name, _id = self.img_and_class[index].split(" ")
        
        if self.phase == "train":
            id_vs_class = {
                i[1]: i[0] for i in [j.split(" ") for j in read_txt(Global.CFG.DATA.IMAGENET_CLASS_VS_ID_TXT)]
            }
            img_path = os.path.join(Global.CFG.DATA.IMAGENET_TRAIN_IMAGES, id_vs_class[_id], img_name)
        elif self.phase == "val":
            img_path = os.path.join(Global.CFG.DATA.IMAGENET_VAL_IMAGES, img_name)
        else:
            Global.LOGGER.error("Invalid phase")

        img = cv2.imread(img_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms: img = self.transforms(img)

        return img, int(_id)
