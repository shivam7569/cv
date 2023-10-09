import cv2
import numpy as np


def readImage(img_path, uint8):
    img = cv2.imread(img_path, -1)

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
    