import os
import torch
import random
from PIL import ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision.transforms.functional import pil_to_tensor

from cv.utils import Global
from cv.src.custom_transforms import *
from cv.utils.file_utils import read_txt
from cv.src import pipeline_functions as PF
from cv.utils import MetaWrapper
from cv.utils.transforms_utils import Augments
from cv.utils.imagenet_utils import ImagenetData


class ClassificationDataset(Dataset, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "ImageNet Dataset Class for training of backbones on classification"

    INVERSE_TRANSFORM = T.Compose(
        [
            T.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
            T.Normalize(
                mean=[-0.485, -0.456, -0.406],
                std=[1., 1., 1.]
            )
        ]
    )

    def __init__(self, phase, transforms=None, debug=None, log=True, ddp=False, standalone=False):

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


        self.transforms = transforms
        if not standalone:
            if transforms is not None:
                self.transforms = T.Compose(self.parseTransforms(transforms))

        self.ddp = ddp
        if ddp:
            self.cfg = Global.CFG

        if Global.CFG.DATA_MIXING.enabled and self.phase == "train":
            self.data_mixing = MixUp(**{k: v for k, v in Global.CFG.DATA_MIXING.items() if k != "enabled"})

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
                                transformations=ClassificationDataset.parseTransforms(transform_params["transformations"]),
                                repeats=transform_params["repeats"],
                                p=transform_params["p"]
                            )
                        else:
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

        if self.phase == "train" and Global.CFG.DATA_MIXING.enabled:
            stacked_images, stacked_labels = self.data_mixing(x=stacked_images, target=stacked_labels)
        if self.phase == "val" and Global.CFG.DATA_MIXING.one_hot_targets:
            stacked_labels = Augments.one_hot(stacked_labels, 1000, on_value=1.0, off_value=0.0)

        return (stacked_images, stacked_labels)

    @staticmethod
    def _vizualizeBatch(batch, k=16):

        imagenet_id_vs_name = ImagenetData.getIdVsName()

        images, labels = batch

        image_size = images.size(-1)

        k = min(k, images.size(0))

        if isinstance(labels, list):
            labels = labels[2] * labels[0] + (1 - labels[2]) * labels[1]
            labels = labels.long()

        random_indices = torch.randperm(images.size(0))

        images, labels = images[random_indices][:k], labels[random_indices][:k].tolist()
        class_names = [imagenet_id_vs_name[str(i)] for i in labels]

        images = ClassificationDataset.INVERSE_TRANSFORM(images)
        out = make_grid(images, nrow=4, padding=2)
        canvas = T.ToPILImage()(out)

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(Global.CFG.FONT_PATH, 20)

        for i in range(4):
            for j in range(k // 4):
                draw.text(((image_size+4) * i + 25, (image_size+4) * (j + 1) - 50), class_names[i+j*4], (0, 102, 204), font=font)

        canvas = pil_to_tensor(canvas)

        return canvas

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
