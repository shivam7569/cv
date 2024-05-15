import random
import cv2
import numpy as np
from PIL import Image

__MASK_PIPELINE_FUNCTIONS__ = [
    "mask_to_img_size"
]


def readImage(img_path, uint8):

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)[:, :, ::-1]

    if len(img.shape) == 2:
        height, width = img.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = img
        rgb_image[:, :, 1] = img
        rgb_image[:, :, 2] = img

        img = rgb_image
    
    if uint8:
        img = img.astype('uint8')

    return img

def resizeWithAspectRatio(img, size):
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
    
    img = cv2.resize(img, (new_w, new_h))

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

def mask_to_img_size(img, mask):
    height, width, _ = img.shape
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return mask