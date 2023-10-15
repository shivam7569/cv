import random
import cv2
import numpy as np
from PIL import Image


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
