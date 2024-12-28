import torch
import random
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision.transforms.functional import pil_to_tensor

from cv.utils import Global
from cv.utils import MetaWrapper
from cv.src.custom_transforms import *
from cv.utils.file_utils import read_txt
from cv.src import pipeline_functions as PF


class ContrastiveDataset(Dataset, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "ImageNet Dataset Class for training of model on contrastive learning"

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

    def __init__(self, phase, transforms=None, same_transforms=True, debug=None, log=True, ddp=False, standalone=False):

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
                if all(isinstance(x, list) for x in transforms):
                    self.transforms = [T.Compose(self.parseTransforms(trnfrms)) for trnfrms in transforms]
                elif same_transforms:
                    self.transforms = [T.Compose(self.parseTransforms(transforms))] * 2

        if log:
            if same_transforms:
                Global.LOGGER.info(f"Augmentations being used during {phase}: {self.transforms[0]}")
            else:
                Global.LOGGER.info(f"Augmentations being used during {phase}: {self.transforms}")

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

        batch, labels = [i[0] for i in batch], [i[1] for i in batch]

        try:
            if Global.CFG.COLLATE_FN.PROCESS == "RandomSize":
                sizes = Global.CFG.COLLATE_FN.SIZES
                batch_sample_spatial_size = random.choice(sizes)

                for transform_set in self.transforms:
                    for trnsfrm in transform_set.transforms:
                        if isinstance(trnsfrm, T.RandomCrop):
                            trnsfrm.size = (batch_sample_spatial_size, batch_sample_spatial_size)
        except Exception as e:
            pass

        if self.transforms is not None:
            processed_batches = [[t(b) for b in batch] for t in self.transforms]

        else:
            processed_batches = batch

        stacked_images = torch.cat([torch.stack(pb, dim=0) for pb in processed_batches], dim=0)
        stacked_labels = torch.tensor(labels)

        return (stacked_images, stacked_labels)

    @staticmethod
    def _vizualizeBatch(images, k=16):

        B, _, _, _ = images.shape
        B //= 2

        indices = []

        for i in range(B):
            indices.extend([i, i+B])

        k = min(k, images.size(0))
        images = images[indices][:k]

        images = ContrastiveDataset.INVERSE_TRANSFORM(images)
        out = make_grid(images, nrow=4, padding=2)
        canvas = T.ToPILImage()(out)

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
                                transformations=ContrastiveDataset.parseTransforms(transform_params["transformations"]),
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
