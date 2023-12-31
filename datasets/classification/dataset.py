import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils.global_params import Global
from utils.file_utils import read_txt
import src.pipeline_functions as PF
from src.custom_transforms import *

class ClassificationDataset(Dataset):

    def __init__(self, phase, transforms=None, debug=None, log=True, ddp=False):

        if log: Global.LOGGER.info(f"Parsing imagenet classification data for {phase}")

        self.phase = phase
        if phase == "train":
            self.phase_text = Global.CFG.DATA.IMAGENET_TRAIN_TXT
        elif phase == "val":
            self.phase_text = Global.CFG.DATA.IMAGENET_VAL_TXT
        else:
            Global.LOGGER.error("Invalid phase")

        self.img_and_class = read_txt(self.phase_text)
        random.shuffle(self.img_and_class)

        if debug is not None:
            self.img_and_class = self.img_and_class[:debug]

        if transforms is not None:
            self.transforms = T.Compose(self.parseTransforms(transforms))
        else: self.transforms = transforms

        self.ddp = ddp
        if ddp:
            self.cfg = Global.CFG

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
                        transform_instnce = globals()[transform_name](**transform_params)
                        transforms_list.append(transform_instnce)

        return transforms_list
    
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

    def collate_fn(self, batch):

        try:
            if Global.CFG.COLLATE_FN.PROCESS == "RandomSize":
                sizes = Global.CFG.COLLATE_FN.SIZES
                batch_sample_spatial_size = random.choice(sizes)

                for trnsfrm in self.transforms.transforms:
                    if isinstance(trnsfrm, T.RandomCrop):
                        trnsfrm.size = (batch_sample_spatial_size, batch_sample_spatial_size)
        except:
            pass

        if self.transforms is not None:
            processed_batch = [(self.transforms(b[0]), b[1]) for b in batch]
        else:
            processed_batch = batch

        images = [pb[0] for pb in processed_batch]
        labels = [pb[1] for pb in processed_batch]

        stacked_images = torch.stack(images, dim=0)
        stacked_labels = torch.tensor(labels)

        return (stacked_images, stacked_labels)

    def __len__(self):
        return len(self.img_and_class)
    
    def __getitem__(self, index):

        if self.ddp:
            Global.CFG = self.cfg

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

        img = self.executePipeline(img_path)

        return img, int(_id)
