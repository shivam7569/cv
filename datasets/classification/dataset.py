import os
import cv2
from torch.utils.data import Dataset
from configs.config import get_cfg

from global_params import Global
from utils.file_utils import read_txt

class ClassificationDataset(Dataset):

    def __init__(self, phase):

        Global.LOGGER.info(f"Parsing imagenet classification data for {phase}")

        self.phase = phase
        if phase == "train":
            self.phase_text = Global.CFG.DATA.IMAGENET_TRAIN_TXT
        elif phase == "val":
            self.phase_text = Global.CFG.DATA.IMAGENET_VAL_TXT
        else:
            Global.LOGGER.error("Invalid phase")

        self.img_and_class = read_txt(self.phase_text)

    def __len__(self):
        return len(self.img_and_class)
    
    def __getitem__(self, index):
        img_name, _id = self.img_and_class[index].split(" ")
        
        if self.phase == "train":
            id_vs_class = {
                i[1]: i[0] for i in [j.split(" ") for j in read_txt(Global.CFG.DATA.IMAGENET_CLASS_VS_ID_TXT)]
            }
            img_path = os.path.join(Global.CFG.DATA.IMAGENET_TRAIN_IMAGES, id_vs_class[_id], img_name)
            img = cv2.imread(img_path, -1)

            return img, _id
        elif self.phase == "val":
            img_path = os.path.join(Global.CFG.DATA.IMAGENET_VAL_IMAGES, img_name)
            img = cv2.imread(img_path, -1)

            return img, _id
        else:
            Global.LOGGER.error("Invalid phase")
