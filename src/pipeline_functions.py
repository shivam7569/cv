import cv2


def readImage(img_path, uint8):
    img = cv2.imread(img_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if uint8: img = img.astype('uint8')

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
    