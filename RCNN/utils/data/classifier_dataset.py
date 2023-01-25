import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm

from RCNN.utils.globalParams import Global

class ClassifierDataset(Dataset):

    def __init__(self, root_dir, transforms=None, phase="train", debug=False):
        super().__init__()

        self.transform = transforms
        self.root_dir = root_dir
        self.phase = phase

        self.samples = list(set([i.split(
            "_")[0] + ".jpg" for i in os.listdir(os.path.join(root_dir, "Annotations"))]))

        if debug: self.samples = random.sample(self.samples, 100)

        self.positive_list = list()
        self.negative_list = list()

        pbar = tqdm(range(len(self.samples)), desc=f"Loading Classifier Data ({phase})", total=len(self.samples))
        for idx in pbar:
            sample_name = self.samples[idx]
            
            positive_annot_path = os.path.join(root_dir, "Annotations", sample_name[:-4] + "_1" + ".json")
            with open(positive_annot_path, 'r') as f:
                positive_annot = json.load(f)
            
            data = np.array(positive_annot["proposal_coord"], dtype=np.int32)
            if data.shape == (0,): continue

            gts = data[:, :-1]
            labels = np.reshape(data[:, -1], newshape=(-1, 1))
            
            if len(gts.shape) == 1:
                if gts.shape[0] == 4:
                    positive_dict = dict()
                    positive_dict["rect"] = gts
                    positive_dict["image_name"] = sample_name
                    positive_dict["image_id"] = idx
                    positive_dict["class"] = labels[0][0]

                    self.positive_list.append(positive_dict)

            else:
                for val in zip(gts, labels):
                    positive_dict = dict()
                    positive_dict["rect"] = val[0]
                    positive_dict["image_name"] = sample_name
                    positive_dict["image_id"] = idx
                    positive_dict["class"] = val[1][0]

                    self.positive_list.append(positive_dict)

            negative_annot_path = os.path.join(root_dir, "Annotations", sample_name[:-4] + "_0" + ".json")
            with open(negative_annot_path, 'r') as f:
                negative_annot = json.load(f)
            
            data = np.array(negative_annot["proposal_coord"], dtype=np.int32)
            if data.shape == (0,): continue

            rects = data[:, :-1]

            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_dict = dict()
                    negative_dict["rect"] = rects
                    negative_dict["image_name"] = sample_name
                    negative_dict["image_id"] = idx
                    negative_dict["class"] = 0

                    self.negative_list.append(negative_dict)

            else:
                for rect in rects:
                    negative_dict = dict()
                    negative_dict["rect"] = rect
                    negative_dict["image_name"] = sample_name
                    negative_dict["image_id"] = idx
                    negative_dict["class"] = 0

                    self.negative_list.append(negative_dict)

    def __getitem__(self, index):

        if index < len(self.positive_list):

            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax = positive_dict["rect"]
            image_name = positive_dict["image_name"]
            image_path = os.path.join(Global.DATA_DIR, self.phase, image_name)
            image = cv2.imread(image_path)
            proposal = image[ymin: ymax, xmin: xmax]
            target = positive_dict["class"]
            cache_dict = positive_dict

        else:

            idx = index - len(self.positive_list)
            negative_dict = self.negative_list[idx]

            xmin, ymin, xmax, ymax = negative_dict["rect"]
            image_name = negative_dict["image_name"]
            image_path = os.path.join(Global.DATA_DIR, self.phase, image_name)
            image = cv2.imread(image_path)
            proposal = image[ymin: ymax, xmin: xmax]
            target = 0
            cache_dict = negative_dict

        if self.transform:
            transformed_proposal = self.transform(image=proposal)
            proposal = transformed_proposal["image"]
        else:
            proposal = cv2.resize(proposal, Global.IMAGE_SIZE)
            proposal = torch.from_numpy(np.moveaxis(proposal, -1, 0)).float()

        return proposal, Global.MAPPED_CLASS_LABELS[target], cache_dict

    def __len__(self):
        return len(self.positive_list) + len(self.negative_list)

    def get_transform(self):
        return self.transform

    def get_images(self):
        return self.samples

    def get_positive_num(self):
        return len(self.positive_list)
    
    def get_negative_num(self):
        return len(self.negative_list)

    def get_positives(self):
        return self.positive_list

    def get_negatives(self):
        return self.negative_list

    def set_negative_list(self, negative_list):
        self.negative_list = negative_list


# data_dir = os.path.join(Global.CLASSIFIER_DATA_DIR, "train")
# dataset = ClassifierDataset(data_dir, None, "train", False)
# print(f"\n Negative proposals: {dataset.get_negative_num()} --- Positive proposals: {dataset.get_positive_num()}")
# a, b, c = dataset.__getitem__(0)
# a, b, c = dataset.__getitem__(108)
# a, b, c = dataset.__getitem__(110)
# a, b, c = dataset.__getitem__(215)
# a = dataset.__len__()
# a = dataset.get_transform()
# a = dataset.get_images()
# a = dataset.get_positives()
# a = dataset.get_negatives()

# print()