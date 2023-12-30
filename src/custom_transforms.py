import cv2
import numpy as np
import torch

class FancyPCA:

    def __init__(self, alpha_std=0.1, p=0.5):
        self.alpha_std = alpha_std
        self.p = p

    def __call__(self, img):

        if self.p < np.random.random():
            return img

        orig_img = img.astype(float).copy()
        img = img / 255.0

        img_rs = img.reshape(-1, 3)

        img_centered = img_rs - np.mean(img_rs, axis=0)
        img_cov = np.cov(img_centered, rowvar=False)

        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        m1 = np.column_stack((eig_vecs))
        m2 = np.zeros((3, 1))

        alpha = np.random.normal(0, self.alpha_std)

        m2[:, 0] = alpha * eig_vals[:]
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):
            orig_img[..., idx] += add_vect[idx]

        orig_img = np.clip(orig_img, 0.0, 255.0)
        orig_img = orig_img.astype(np.uint8)

        return orig_img

class DetectionHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        img, bboxes, anns = data
        _, img_w = img.shape[:2]

        if self.p < np.random.random():
            return img, bboxes, anns

        img = cv2.flip(img, flipCode=1)
        bboxes[:, [0, 2]] = img_w - bboxes[:, [2, 0]]
        anns[:, [0, 2]] = img_w - anns[:, [2, 0]]

        return img, bboxes, anns

def mixup_data(data, targets, alpha, use_cuda=True):

    if use_cuda:
        indices = torch.randperm(data.size(0)).to(data.device)
    else:
        indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    
    return new_data, targets, shuffled_targets, lam

def cutmix_data(data, targets, alpha, use_cuda=True):

    if use_cuda:
        indices = torch.randperm(data.size(0)).to(data.device)
    else:
        indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = shuffled_data[:, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    
    return new_data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(int)
    cut_h = (H * cut_rat).astype(int)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
