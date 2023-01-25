import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from RCNN.utils.globalParams import Global

class HardNegativeMiningDataset(Dataset):

    def __init__(self, negative_list, jpeg_images, transform, phase):
        self.negative_list = negative_list
        self.jpeg_images = jpeg_images
        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        target = 0

        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']
        image_id = negative_dict['image_id']

        image_name = self.jpeg_images[image_id]
        image_path = os.path.join(Global.DATA_DIR, self.phase, image_name)
        image = cv2.imread(image_path)
        proposal = image[ymin: ymax, xmin: xmax]

        if self.transform:
            transformed_proposal = self.transform(image=proposal)
            proposal = transformed_proposal["image"]
        else:
            proposal = cv2.resize(proposal, Global.IMAGE_SIZE)
            proposal = torch.from_numpy(np.moveaxis(proposal, -1, 0)).float()

        return proposal, target, negative_dict
    
    def __len__(self):
        return len(self.negative_list)
    
    def add_hard_negatives(self, hard_negative_list, negative_list, add_negative_list):
        for item in hard_negative_list:
            if len(add_negative_list) == 0 or list(item["rect"]) not in add_negative_list:
                negative_list.append(item)
                add_negative_list.append(list(item["rect"]))

    def get_hard_negatives(self, preds, cache_dicts):
        false_positive_mask = preds != 0
        true_negative_mask = preds == 0

        hard_negative_list = []
        easy_negative_list = []

        for idx in range(len(false_positive_mask)):
            if false_positive_mask[idx] == True:
                entry = {
                    "rect": cache_dicts["rect"][idx].numpy(),
                    "image_id": cache_dicts["image_id"][idx].numpy(),
                    "image_name": cache_dicts["image_name"][idx],
                    "class": cache_dicts["class"][idx].numpy()
                }
                hard_negative_list.append(entry)

        for idx in range(len(true_negative_mask)):
            if true_negative_mask[idx] == True:
                entry = {
                    "rect": cache_dicts["rect"][idx].numpy(),
                    "image_id": cache_dicts["image_id"][idx].numpy(),
                    "image_name": cache_dicts["image_name"][idx],
                    "class": cache_dicts["class"][idx].numpy()
                }
                easy_negative_list.append(entry)

        return hard_negative_list, easy_negative_list