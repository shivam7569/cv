import math
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as T
from utils.transforms_utils import Augmentations

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

class Identity:

    def __init__(self):
        pass

    def __call__(self, img):
        return img

class RandomAugmentation:

    def __init__(
            self,
            m=Augmentations._LEVEL_DENOM,
            n=2,
            p=0.5,
            mstd=0,
            mmax=Augmentations._LEVEL_DENOM,
            increasing=False,
            choice_weights=None
    ):
        
        parameters = dict()
        
        if mstd is not None and mstd > 100: mstd = float("inf")
        parameters.setdefault("magnitude_std", mstd)
        parameters.setdefault("magnitude_max", mmax)
        parameters.setdefault("num_layers", n)

        transforms = Augmentations.getTransforms(increasing=increasing)

        self.ra_ops = Augmentations.getRandomAugmentOps(
            magnitude=m,
            prob=p,
            transforms=transforms,
            parameters=parameters
        )
        self.n = n
        self.choice_weights = choice_weights

    def __call__(self, img):
        ops = np.random.choice(
            self.ra_ops,
            self.n,
            replace=self.choice_weights is None,
            p=self.choice_weights
        )
        for op in ops:
            img = op(img)
        
        return img
    
    def __repr__(self):
        fs = self.__class__.__name__ + f"(n={self.n}, ops="
        for op in self.ra_ops:
            fs += f"\n\t{op}"
        fs += ')'

        return fs


class MixUp:

    def __init__(
            self,
            mixup_alpha=1.0,
            cutmix_alpha=0.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            correct_lam=True,
            label_smoothing=0.0,
            one_hot_targets=False,
            num_classes=1000
    ):
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        self.mixup_enabled = True
        self.one_hot_targets = one_hot_targets

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = Augmentations.cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = Augmentations.mixup_target(target, self.num_classes, lam, self.label_smoothing, self.one_hot_targets)
        
        return x, target

class RandomCutOut:
    
    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1/3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == "rand":
            self.rand_color = True
        elif self.mode == "pixel":
            self.per_pixel = True
        else:
            assert not self.mode or self.mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = Augmentations._get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input_):
        if len(input_.size()) == 3:
            self._erase(input_, *input_.size(), input_.dtype)
        else:
            batch_size, chan, img_h, img_w = input_.size()
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input_[i], chan, img_h, img_w, input_.dtype)
        return input_

    def __repr__(self):
        fs = self.__class__.__name__ + f"(p={self.probability}, mode={self.mode}"
        fs += f", count=({self.min_count}, {self.max_count}))"
        return fs

class ThreeAugment:

    def __init__(self, p=0.5):

        self.p = p
        self.grayscale = T.Grayscale(num_output_channels=3)
        self.solarize = T.RandomSolarize(
            threshold=min(256, int((random.choice(range(10)) / 10.0) * 256)),
            p=1.0
        )
        self.gaussian_blur = T.GaussianBlur(
            kernel_size=(5, 9), sigma=(0.1, 5.0)
        )

    def __call__(self, x):
        if random.random() > self.p:
            return x

        transform = random.choice([self.grayscale, self.solarize, self.gaussian_blur])
        x = transform(x)

        self.solarize.threshold = min(256, int((random.choice(range(10)) / 10.0) * 256))

        return x