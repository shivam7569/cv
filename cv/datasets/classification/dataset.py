import torch
import random
from PIL import ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision.transforms.functional import pil_to_tensor

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.src.custom_transforms import *
from cv.utils.file_utils import read_txt
from cv.src import pipeline_functions as PF
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

        if debug is not None:
            if debug < len(self.img_and_class):
                self.img_and_class = random.sample(self.img_and_class, k=debug)
                
        self.transforms = transforms
        if not standalone:
            if transforms is not None:
                self.transforms = T.Compose(self.parseTransforms(transforms))

        if log: Global.LOGGER.info(f"Augmentations being used during {phase}: {self.transforms}")

        self.ddp = ddp
        if ddp:
            self.cfg = Global.CFG

        if Global.CFG.DATA_MIXING.enabled and self.phase == "train":
            self.data_mixing = MixUp(**{k: v for k, v in Global.CFG.DATA_MIXING.items() if k != "enabled"})
    
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
        random_indices = random.sample([i for i in range(images.size(0))], k=k)

        if isinstance(labels, list):
            lam = labels[-1]
            labels = [(labels[0][j], labels[1][j]) for j in range(labels[0].shape[0])]
            labels = [labels[i] for i in random_indices]
            class_names = [
                f"{imagenet_id_vs_name[str(i[0].item())]}: {round(lam * 100, 1)}%\n{imagenet_id_vs_name[str(i[1].item())]}: {round((1-lam) * 100, 1)}%" 
                for i in labels
            ]
            x_offset, y_offset, font_size = 10, 60, 15
        else:
            labels = labels[torch.tensor(random_indices)].tolist()
            class_names = [imagenet_id_vs_name[str(i)] for i in labels]
            x_offset, y_offset, font_size = 25, 50, 20

        images = images[random_indices]

        images = ClassificationDataset.INVERSE_TRANSFORM(images)
        out = make_grid(images, nrow=4, padding=2)
        canvas = T.ToPILImage()(out)

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(Global.CFG.FONT_PATH, font_size)

        for i in range(4):
            for j in range(k // 4):
                draw.text(((image_size+4) * i + x_offset, (image_size+4) * (j + 1) - y_offset), class_names[i+j*4], (0, 102, 204), font=font)

        canvas = pil_to_tensor(canvas)

        return canvas

    @staticmethod
    def parseTransforms(transforms):
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

    def __len__(self):
        return len(self.img_and_class)
    
    def __getitem__(self, index):

        if self.ddp:
            Global.CFG = self.cfg

        img_path, _id = self.img_and_class[index].split(" ")

        img = self.executePipeline(img_path)

        return img, int(_id)
