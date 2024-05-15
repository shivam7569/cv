import os
import random
import cv2
import numpy as np
import torch
from datasets.classification.dataset import ClassificationDataset
import src.pipeline_functions as PF
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid

from PIL import ImageDraw
from datasets.coco_parser import CocoParser
from utils.global_params import Global
from utils.mask_utils import getColorMask, getOverlay

class SegmentationDataset(Dataset):

    def __init__(self, phase, transforms, ddp, debug=None, log=True):

        if log: Global.LOGGER.info(f"Parsing coco segmentation data for {phase}")

        self.coco_parser = CocoParser(phase=phase)
        if phase == "train":
            self.img_dir = Global.CFG.DATA.COCO_TRAIN_IMAGES_DIR
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

    def collate_fn(self, batch):

        if self.transforms is not None:
            processed_batch = [(self.transforms((b[0][:, :, ::-1], b[1]))) for b in batch]
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
                else:
                    img = getattr(PF, function["func"])(img, **function["params"])
        else:
            Global.LOGGER.error("Cannot proceed without data pipeline")

        return img, mask
    
    @staticmethod
    def _vizualizeBatch(batch, k=4):

        images, masks = batch
        images, masks = images.cpu(), masks.cpu()

        k = min(k, images.size(0))

        random_indices = torch.randperm(images.size(0))

        images, masks = images[random_indices][:k], masks[random_indices][:k]
        images = ClassificationDataset.INVERSE_TRANSFORM(images)

        overlayed_batch = []
        for i in range(k):
            img, mask = (images[i] * 255).to(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1], masks[i].to(torch.uint8).squeeze(0).numpy()
            color_mask = getColorMask(mask)
            overlay_image = getOverlay(img, color_mask)

            overlayed_batch.append(torch.from_numpy(overlay_image[:, :, ::-1].copy()).permute(2, 0, 1))

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
