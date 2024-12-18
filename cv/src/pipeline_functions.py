import cv2
import random
import numpy as np
from PIL import Image

from cv.utils import Global

__MASK_PIPELINE_FUNCTIONS__ = [
    "mask_to_img_size"
]

__COMBINED_PIPELINE_FUNCTIONS__ = [
    "remove_bg",
    "fit_to_size"
]


def readImage(img_path, uint8, rgb=True):

    img = Image.open(img_path).convert("RGB") if rgb else Image.open(img_path)
    img = np.array(img)

    if len(img.shape) == 2 and rgb:
        height, width = img.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = img
        rgb_image[:, :, 1] = img
        rgb_image[:, :, 2] = img

        img = rgb_image
    elif not rgb:
        img = img[..., np.newaxis]
    
    if uint8:
        img = img.astype('uint8')

    return img

def resizeWithAspectRatio(img, size, interpolation="bilinear"):
    # aspect_ratio = h/w
    img_h, img_w = img.shape[:2]
    aspect_ratio = img_h / img_w

    if img_h < img_w:

        if img_h == size: return img

        new_h = size
        new_w = int(new_h / aspect_ratio)
    else:

        if img_w == size: return img
        
        new_w = size
        new_h = int(new_w * aspect_ratio)
    
    if interpolation == "bilinear":
        interpolation_method = cv2.INTER_CUBIC
    if interpolation == "nearest":
        interpolation_method = cv2.INTER_NEAREST
    
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)

    return img

def alexNetresize(img):

    # aspect_ratio = h/w
    img_h, img_w = img.shape[:2]
    aspect_ratio = img_h / img_w

    if img_h < img_w:
        new_h = 256
        new_w = int(new_h / aspect_ratio)
    else:
        new_w = 256
        new_h = int(new_w * aspect_ratio)
    
    img = cv2.resize(img, (new_w, new_h))

    return img

def vgg16resize(img, s_min, s_max):
    scales = range(s_min, s_max + 1)
    training_scale = random.choice(scales)

    # aspect_ratio = h/w
    img_h, img_w = img.shape[:2]
    aspect_ratio = img_h / img_w

    if img_h < img_w:
        new_h = training_scale
        new_w = int(new_h / aspect_ratio)
    else:
        new_w = training_scale
        new_h = int(new_w * aspect_ratio)

    img = cv2.resize(img, (new_w, new_h))

    return img

def inceptionv1resize(img, aspect_ratio_min, aspect_ratio_max):

    aspect_ratio = random.uniform(aspect_ratio_min, aspect_ratio_max)

    # aspect_ratio = w/h
    img_h, img_w = img.shape[:2]

    if img_h < img_w:
        new_h = 256
        new_w = int(new_h * aspect_ratio)
        if new_w < 224: new_w = 224
    else:
        new_w = 256
        new_h = int(new_w / aspect_ratio)
        if new_h < 224: new_h = 224

    interpolation_methods = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC]
    interpolation_method = random.choice(interpolation_methods)

    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)

    return img

def inceptionv2resize(img, aspect_ratio_min, aspect_ratio_max):

    aspect_ratio = random.uniform(aspect_ratio_min, aspect_ratio_max)

    # aspect_ratio = w/h
    img_h, img_w = img.shape[:2]

    if img_h < img_w:
        new_h = 340
        new_w = int(new_h * aspect_ratio)
        if new_w < 299: new_w = 299
    else:
        new_w = 340
        new_h = int(new_w / aspect_ratio)
        if new_h < 299: new_h = 299

    interpolation_methods = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC]
    interpolation_method = random.choice(interpolation_methods)

    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)

    return img

def extractROI(img, roi_bbox, dilated=0):

    x1, y1, x2, y2 = roi_bbox

    dilated_x1 = max(x1 - dilated, 0)
    dilated_y1 = max(y1 - dilated, 0)
    dilated_x2 = min(x2 + dilated, img.shape[1])
    dilated_y2 = min(y2 + dilated, img.shape[0])

    roi = img[dilated_y1: dilated_y2, dilated_x1: dilated_x2]

    return roi

def rcnn_warpROI(roi, size):
    scale_x = size[0] / roi.shape[1]
    scale_y = size[1] / roi.shape[0]

    affine_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)

    warped_roi = cv2.warpAffine(roi, affine_matrix, size)

    return warped_roi

def fast_rcnn_resize(img, min_size, max_size):
    img = resizeWithAspectRatio(img=img, size=min_size)
    img_h, img_w = img.shape[:2]

    aspect_ratio = img_h / img_w

    if img_h > max_size:
        new_h = max_size
        new_w = int(new_h / aspect_ratio)

        shrinking = (new_w * new_h) / (img_w * img_h) < 1
        interpolation_method = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC

        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)
    elif img_w > max_size:
        new_w = max_size
        new_h = int(new_w * aspect_ratio)

        shrinking = (new_w * new_h) / (img_w * img_h) < 1
        interpolation_method = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC

        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation_method)

    return img

def scale_factor_resize(img, scales):
    scale = random.choice(scales)

    H, W, _ = img.shape
    img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_LINEAR)

    return img

def mask_to_img_size(img, mask):
    height, width, _ = img.shape
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return mask

def img_resize(img, size):
    img = cv2.resize(img, dsize=size)

    return img

def remove_bg(img, mask, threshold=5, size=256):

    clamp = lambda x, y, z: max(min(z, x), y)

    H, W = mask.shape
    top, bottom, left, right = 0, 0, 0, 0
    for i in range(H):
        row = mask[i, :]
        if np.unique(row).shape == (1,) and np.unique(row)[0] == 0:
            top += 1
        else:
            break

    for i in range(H-1, -1, -1):
        row = mask[i, :]
        if np.unique(row).shape == (1,) and np.unique(row)[0] == 0:
            bottom += 1
        else:
            break

    for i in range(W):
        col = mask[:, i]
        if np.unique(col).shape == (1,) and np.unique(col)[0] == 0:
            left += 1
        else:
            break

    for i in range(W-1, -1, -1):
        col = mask[:, i]
        if np.unique(col).shape == (1,) and np.unique(col)[0] == 0:
            right += 1
        else:
            break

    top = clamp(top - threshold, 0, H)
    bottom = clamp(bottom - threshold, 0, H)
    right = clamp(right - threshold, 0, W)
    left = clamp(left - threshold, 0, W)

    new_height = clamp(H - top - bottom, 0, H)
    new_width = clamp(W - right - left, 0, W)

    diff_height = max(0, size - new_height)
    new_h_threshold = threshold + (diff_height // 2)

    diff_width = max(0, size - new_width)
    new_w_threshold = threshold + (diff_width // 2)

    y1 = clamp(top-new_h_threshold, 0, H)
    y2 = clamp(H-(bottom-new_h_threshold), 0, H)
    x1 = clamp(left-new_w_threshold, 0, W)
    x2 = clamp(W-(right-new_w_threshold), 0, W)

    clean_mask = mask[y1: y2, x1: x2]
    clean_img = img[y1: y2, x1: x2]

    return clean_img, clean_mask

def fit_to_size(img, mask, size, padding, ignore_label=-1):
    H, W, _ = img.shape

    if padding:
        height_diff = max(0, size - H)
        width_diff = max(0, size - W)

        padding_kwargs = {
            "top": height_diff // 2,
            "bottom": height_diff - (height_diff // 2),
            "left": width_diff // 2,
            "right": width_diff - (width_diff // 2),
            "borderType": cv2.BORDER_CONSTANT
        }

        if height_diff > 0 or width_diff > 0:
            img = cv2.copyMakeBorder(img, value=Global.CFG.COCO_MEAN, **padding_kwargs)
            mask = cv2.copyMakeBorder(mask, value=ignore_label, **padding_kwargs)

    else:
        img = resizeWithAspectRatio(img=img, interpolation="bilinear")
        mask = resizeWithAspectRatio(img=mask, interpolation="nearest")

    return img, mask