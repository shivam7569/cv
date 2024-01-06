from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
import os
from threading import Lock
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import polars as pl
from torch.utils.data import Dataset
from src.custom_transforms import DetectionHorizontalFlip
from datasets.coco_parser import CocoParser

from datasets.detection.dataset import DetectionDataset
from src.metrics import DetectionMetrics
from utils.global_params import Global

import torchvision.transforms as T

class FastRCNNData(DetectionDataset):

    def __init__(self, phase, debug=None):
        super(FastRCNNData, self).__init__(phase=phase)

        self.phase = phase

        phase_positives_path = os.path.join(Global.CFG.FAST_RCNN.RCNN_ROI_CSV_DIR, f"{phase}_positive.csv")
        phase_negatives_path = os.path.join(Global.CFG.FAST_RCNN.RCNN_ROI_CSV_DIR, f"{phase}_negative.csv")

        self.positive_df = pl.read_csv(phase_positives_path)
        self.negative_df = pl.read_csv(phase_negatives_path)

        if debug is not None:
            self.positive_df = self.positive_df.head(n=debug)
            self.negative_df = self.negative_df.head(n=debug)

        self.positive_df = self.positive_df.with_columns(pl.col("roi_x1").cast(pl.Int32, strict=True).alias("roi_x1"))
        self.positive_df = self.positive_df.with_columns(pl.col("roi_y1").cast(pl.Int32, strict=True).alias("roi_y1"))
        self.positive_df = self.positive_df.with_columns(pl.col("roi_x2").cast(pl.Int32, strict=True).alias("roi_x2"))
        self.positive_df = self.positive_df.with_columns(pl.col("roi_y2").cast(pl.Int32, strict=True).alias("roi_y2"))
        self.positive_df = self.positive_df.with_columns(pl.col("roi_class").cast(pl.Int32, strict=True).alias("roi_class"))

        self.negative_df = self.negative_df.with_columns(pl.col("roi_x1").cast(pl.Int32, strict=True).alias("roi_x1"))
        self.negative_df = self.negative_df.with_columns(pl.col("roi_y1").cast(pl.Int32, strict=True).alias("roi_y1"))
        self.negative_df = self.negative_df.with_columns(pl.col("roi_x2").cast(pl.Int32, strict=True).alias("roi_x2"))
        self.negative_df = self.negative_df.with_columns(pl.col("roi_y2").cast(pl.Int32, strict=True).alias("roi_y2"))
        self.negative_df = self.negative_df.with_columns(pl.col("roi_class").cast(pl.Int32, strict=True).alias("roi_class"))

    def create_positive_roi_data(self):

        return # safe check to avoid recreation of data, to be removed at last

        positive_df_path = os.path.join(Global.CFG.FAST_RCNN.ROI_CSV_DIR, f"{self.phase}_positive.csv")
        self.positive_df.write_csv(positive_df_path)

    @classmethod
    def roi_gts_iou(cls, row, gts):
        roi = [row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"]]
        ious = DetectionMetrics.iou(roi, gts)

        assert len(ious) == gts.shape[0]

        positive = any(0.1 <= element < 0.5 for element in ious)

        return positive

    def process_negative_df(self, i, negative_csv_writer, lock):

        Global.LOGGER.info(f"Processing image with id: {i}")

        data = super().__getitem__(i, return_dims=True, return_img_path=True)
        _, ann, dims, img_path = data["img"], data["annotations"], data["dims"], data["img_path"]

        if ann.shape[0] == 0: return
        img_w, img_h = dims["img_w"], dims["img_h"]

        ann[:, 0] = ann[:, 0] * img_w
        ann[:, 1] = ann[:, 1] * img_h
        ann[:, 2] = ann[:, 2] * img_w
        ann[:, 3] = ann[:, 3] * img_h

        gts = ann[:, :-1].astype(int)

        roi_df = self.negative_df.filter(self.negative_df["image_path"] == img_path)
        roi_df = roi_df.with_columns(pl.struct(pl.col('*')).map_elements(
            lambda x: FastRCNNData.roi_gts_iou(x, gts), skip_nulls=False, return_dtype=pl.Boolean).alias("flag")
        )
        roi_df = roi_df.filter(roi_df["flag"] == True).drop("flag")

        with lock:
            for row in roi_df.rows(named=True):
                row = [row["image_path"], row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"], row["roi_class"]]
                negative_csv_writer.writerow(row)

    def create_negative_roi_data(self):
        negative_df_path = os.path.join(Global.CFG.FAST_RCNN.ROI_CSV_DIR, f"{self.phase}_negative.csv")
        
        columns = ["image_path", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_class"]

        with open(negative_df_path, "w", newline='') as negative_file:
            negative_csv_writer = csv.writer(negative_file)

            negative_csv_writer.writerow(columns)
            
            lock = Lock()

            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(len(self.coco_parser.imgIds)):
                    futures.append(executor.submit(self.process_negative_df, i, negative_csv_writer, lock))

                for future in as_completed(futures):
                    future.result()

class FastRCNNDataset(DetectionDataset):

    def __init__(self, phase, transforms=None, debug=None):
        super(FastRCNNDataset, self).__init__(phase=phase, debug=debug)

        positive_roi_csv_path = os.path.join(Global.CFG.FAST_RCNN.ROI_CSV_DIR, f"{phase}_positive.csv")
        negative_roi_csv_path = os.path.join(Global.CFG.FAST_RCNN.ROI_CSV_DIR, f"{phase}_negative.csv")

        self.positive_roi_df = pl.read_csv(positive_roi_csv_path)
        self.negative_roi_df = pl.read_csv(negative_roi_csv_path)

        if transforms is not None:
            self.transforms = T.Compose(self.parseTransforms(transforms))
        else: self.transforms = transforms

        self._cleanData()

    def _cleanData(self):

        common_roi_images = set(self.positive_roi_df["image_path"].unique()).intersection(set(self.negative_roi_df["image_path"].unique()))
        path_lambda = lambda i: os.path.join(self.img_dir, self.coco_parser.getImgInfo(img_id=i)["filename"])

        self.coco_parser.imgIds = [i for i in self.coco_parser.imgIds if path_lambda(i) in common_roi_images]

    def __len__(self):
        return super().__len__()

    def _getAssociatedGT(self, roi, gts):
        ious = DetectionMetrics.iou(roi, gts)
        assert len(ious) == gts.shape[0]

        corres_gt = gts[ious.index(max(ious))]

        return corres_gt

    def _parameterizeBboxTargets(self, rois, gts):

        parameterized_targets = []
        for roi in rois:
            if roi[-1] == 0:
                parameterized_targets.append([-1.0, -1.0, -1.0, -1.0])
            else:

                asso_gt = self._getAssociatedGT(roi=roi[:-1], gts=gts[:, :-1])

                p_x1, p_y1, p_x2, p_y2, _ = roi

                g_x1, g_y1, g_x2, g_y2 = asso_gt

                p_x = (p_x1 + p_x2) // 2
                p_y = (p_y1 + p_y2) // 2
                p_w = p_x2 - p_x1
                p_h = p_y2 - p_y1

                g_x = (g_x1 + g_x2) // 2
                g_y = (g_y1 + g_y2) // 2
                g_w = g_x2 - g_x1
                g_h = g_y2 - g_y1

                t_x = (g_x - p_x) / p_w
                t_y = (g_y - p_y) / p_h
                t_w = math.log(g_w / p_w)
                t_h = math.log(g_h / p_h)

                parameterized_targets.append([t_x, t_y, t_w, t_h])

        parameterized_targets = np.array(parameterized_targets)

        return parameterized_targets

    def _getROIs(self, img_path, img_area):

        positive_roi_df = self.positive_roi_df.filter(self.positive_roi_df["image_path"] == img_path)
        negative_roi_df = self.negative_roi_df.filter(self.negative_roi_df["image_path"] == img_path)

        negative_roi_df = negative_roi_df.with_columns(((pl.col("roi_x2") - pl.col("roi_x1")) * (pl.col("roi_y2") - pl.col("roi_y1"))).alias("area"))
        negative_roi_df = negative_roi_df.with_columns(pl.lit(img_area).alias("img_area"))
        negative_roi_df = negative_roi_df.with_columns((pl.col("area") / pl.col("img_area")).alias("ratio"))
        negative_roi_df = negative_roi_df.filter(negative_roi_df["ratio"] < 0.70).drop(["area", "img_area", "ratio"])

        positive_roi_df_replacement = False if positive_roi_df.shape[0] >= Global.CFG.FAST_RCNN.IMAGE_POSITIVE_ROI_BATCH_SIZE else True
        negative_roi_df_replacement = False if negative_roi_df.shape[0] >= Global.CFG.FAST_RCNN.IMAGE_NEGATIVE_ROI_BATCH_SIZE else True
        
        positive_rois = positive_roi_df.sample(n=Global.CFG.FAST_RCNN.IMAGE_POSITIVE_ROI_BATCH_SIZE, shuffle=True, with_replacement=positive_roi_df_replacement)
        negative_rois = negative_roi_df.sample(n=Global.CFG.FAST_RCNN.IMAGE_NEGATIVE_ROI_BATCH_SIZE, shuffle=True, with_replacement=negative_roi_df_replacement)

        positive_rois = positive_rois[["roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_class"]].to_numpy().astype(float)
        negative_rois = negative_rois[["roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi_class"]].to_numpy().astype(float)

        return positive_rois, negative_rois

    def _applyTransforms(self, img, rois, anns):
        for transform in self.transforms.transforms:
            if isinstance(transform, DetectionHorizontalFlip):
                img, rois, anns = transform([img, rois, anns])
            else:
                img = transform(img)

        return img, rois, anns

    def _getBatchImagestoSameSize(self, batch):
        height = max([i["img"].shape[0] for i in batch])
        width = max([i["img"].shape[1] for i in batch])

        for i in batch:
            i["img"] = cv2.resize(i["img"], (width, height))

        return batch

    def verify(self, img, rois, anns):
        positive_rois = rois[:Global.CFG.FAST_RCNN.IMAGE_POSITIVE_ROI_BATCH_SIZE, :]
        negative_rois = rois[Global.CFG.FAST_RCNN.IMAGE_POSITIVE_ROI_BATCH_SIZE:Global.CFG.FAST_RCNN.IMAGE_NEGATIVE_ROI_BATCH_SIZE, :]

        for pb in positive_rois.astype(int):
            x1, y1, x2, y2 = pb[:4]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

        for pb in negative_rois.astype(int):
            x1, y1, x2, y2 = pb[:4]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))

        for pb in anns.astype(int):
            x1, y1, x2, y2, c = pb
            name = CocoParser.getIdVsName()[str(c-1)]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            img = cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite("workshop/Sample.png", img)


    def collate_fn(self, batch):

        batch = self._getBatchImagestoSameSize(batch)

        images = []
        region_of_interest = []

        for batch_entry in batch:
            img = batch_entry["img"]
            img_h, img_w = img.shape[:2]

            rois = batch_entry["rois"]
            annots = batch_entry["annotations"]

            rois[:, [0, 2]] = rois[:, [0, 2]] * img_w
            rois[:, [1, 3]] = rois[:, [1, 3]] * img_h

            annots[:, [0, 2]] = annots[:, [0, 2]] * img_w
            annots[:, [1, 3]] = annots[:, [1, 3]] * img_h

            img, rois, annots = self._applyTransforms(img, rois, annots)

            prmtrzd_trgts = self._parameterizeBboxTargets(rois.astype(int), annots.astype(int))
            assert prmtrzd_trgts.shape[0] == rois.shape[0]
            rois = np.concatenate([rois.astype(int), prmtrzd_trgts], axis=1)

            rois[:, [0, 2]] = rois[:, [0, 2]] / img_w
            rois[:, [1, 3]] = rois[:, [1, 3]] / img_h

            images.append(img)
            np.random.shuffle(rois)
            region_of_interest.append(rois)

        images = torch.stack(images, dim=0)
        # region_of_interest = torch.stack([torch.from_numpy(i) for i in region_of_interest], dim=0)
        region_of_interest = np.stack(region_of_interest, axis=0)

        return (images, region_of_interest)
    
    def __getitem__(self, idx):
        data = super().__getitem__(index=idx, return_dims=True, return_img_path=True)

        img = data["img"]
        img_path = data["img_path"]
        img_w, img_h = data["dims"]["img_w"], data["dims"]["img_h"]

        anns = data["annotations"]
        anns[:, -1] += 1

        positive_rois, negative_rois = self._getROIs(img_path=img_path, img_area=img_w*img_h)

        positive_rois[:, [0, 2]] = positive_rois[:, [0, 2]] / img_w
        positive_rois[:, [1, 3]] = positive_rois[:, [1, 3]] / img_h

        negative_rois[:, [0, 2]] = negative_rois[:, [0, 2]] / img_w
        negative_rois[:, [1, 3]] = negative_rois[:, [1, 3]] / img_h

        rois = np.concatenate([positive_rois, negative_rois], axis=0)

        cv2.imwrite("workshop/Example.png", img)

        return_data = {
            "img": img,
            "rois": rois,
            "annotations" : anns
        }

        return return_data
