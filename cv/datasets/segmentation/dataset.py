import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid

from cv.utils import Global
from cv.src import pipeline_functions as PF
from cv.datasets import ClassificationDataset
from cv.datasets.coco_parser import CocoParser
from cv.utils import MetaWrapper
from cv.utils.mask_utils import getColorMask, getOverlay

class SegmentationDataset(Dataset, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Coco Dataset Class for training of models on segmentation"

    INVERSE_TRANSFORM = T.Compose(
        [
            T.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1/0.278, 1/0.274, 1/0.289]
            ),
            T.Normalize(
                mean=[-0.470, -0.447, -0.408],
                std=[1., 1., 1.]
            )
        ]
    )

    def __init__(self, phase, transforms, ddp, debug=None, log=True):

        if log: Global.LOGGER.info(f"Parsing coco segmentation data for {phase}")

        self.coco_parser = CocoParser(phase=phase)
        if phase == "train":
            self.img_dir = Global.CFG.DATA.COCO_TRAIN_IMAGES_DIR
            self._clean()
        elif phase == "val":
            self.img_dir = Global.CFG.DATA.COCO_VAL_IMAGES_DIR
        else:
            Global.LOGGER.error("Invalid phase")

        self.phase = phase

        self.transforms = transforms
        if transforms is not None:
            self.transforms = T.Compose(ClassificationDataset.parseTransforms(transforms))

        self.ddp = ddp
        if ddp:
            self.cfg = Global.CFG

        if debug is not None:
            self.coco_parser.imgIds = random.sample(self.coco_parser.imgIds, k=debug)

    def _clean(self):
        exclude_ids = self.coco_parser.getExcludeIDs()
        self.coco_parser.imgIds = [i for i in self.coco_parser.imgIds if i not in exclude_ids]

    def collate_fn(self, batch):

        if self.transforms is not None:
            processed_batch = [(self.transforms((b[0], b[1]))) for b in batch]
        else:
            processed_batch = batch

        images = [pb[0] for pb in processed_batch]
        masks = [pb[1] for pb in processed_batch]

        stacked_images = torch.stack(images, dim=0)
        stacked_masks = torch.stack(masks, dim=0)

        return (stacked_images, stacked_masks)

    def executePipeline(self, img_id):

        img_info = self.coco_parser.getImgInfo(img_id=img_id)
        img_path = os.path.join(self.img_dir, img_info["filename"])

        mask = self.coco_parser.getSegmentationMask(img_id=img_id)

        img_pipeline = Global.CFG.PIPELINES.TRAIN if self.phase == "train" else Global.CFG.PIPELINES.VAL
        if img_pipeline is not None:
            for function in img_pipeline:
                if function["func"] == "readImage":
                    img = getattr(PF, function["func"])(img_path, **function["params"])
                elif function["func"] in PF.__MASK_PIPELINE_FUNCTIONS__:
                    mask = getattr(PF, function["func"])(img, mask, **function["params"])
                elif function["func"] in PF.__COMBINED_PIPELINE_FUNCTIONS__:
                    img, mask = getattr(PF, function["func"])(img, mask, **function["params"])
                else:
                    img = getattr(PF, function["func"])(img, **function["params"])
        else:
            Global.LOGGER.error("Cannot proceed without data pipeline")

        return img, mask
    
    @staticmethod
    def get_class_weights(loader, method, rank, normalize=True):

        loader_iter = tqdm(loader, desc=f"Calculating {method.split('_')[0]} class frequencies", unit="batch") if not rank else loader

        frequencies = {
            i: 0 for i in CocoParser.getIdVsName().keys()
        }
        for batch in loader_iter:
            _, masks = batch
            classes, class_counts = np.unique(masks.numpy(), return_counts=True)
            
            for i in range(classes.shape[0]):
                try:
                    frequencies[str(classes[i])] += class_counts[i]
                except KeyError:
                    pass

        if 0 in list(frequencies.values()):
            Global.LOGGER.warn(f"While calculating class frequencies, some classes were not found in dataset")

        total_class_frequencies = np.array(list(frequencies.values()), dtype=np.longlong)
        total_class_frequencies[total_class_frequencies == 0] = 1

        if method == "inverse_frequency":
            weights = 1 / total_class_frequencies
            if normalize: weights /= np.sum(weights)
        elif method == "median_frequency":
            class_median = np.median(total_class_frequencies)
            weights = class_median / total_class_frequencies
            if normalize: weights /= np.sum(weights)

        weights = torch.from_numpy(weights).to(torch.float32)
        
        return weights
    
    @staticmethod
    def _vizualizeBatch(batch, k=4):

        images, masks = batch
        images, masks = images.cpu(), masks.cpu()

        k = min(k, images.size(0))

        random_indices = torch.randperm(images.size(0))

        images, masks = images[random_indices][:k], masks[random_indices][:k]
        images = SegmentationDataset.INVERSE_TRANSFORM(images)

        overlayed_batch = []
        for i in range(k):
            img, mask = (images[i] * 255).to(torch.uint8).permute(1, 2, 0).numpy(), masks[i].to(torch.uint8).squeeze(0).numpy()
            color_mask = getColorMask(mask)
            overlay_image = getOverlay(img, color_mask)
            overlayed_batch.append(torch.from_numpy(overlay_image.copy()).permute(2, 0, 1))

        overlayed_batch = torch.stack(overlayed_batch, dim=0)

        canvas = make_grid(overlayed_batch, nrow=2, padding=2)

        return canvas

    def __len__(self):
        return len(self.coco_parser.imgIds)
    
    def __getitem__(self, index):
        if self.ddp:
            Global.CFG = self.cfg

        img_id = self.coco_parser.imgIds[index]

        img, mask = self.executePipeline(img_id=img_id)

        return img, mask
