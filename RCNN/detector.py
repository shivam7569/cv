import os
import random
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pycocotools.coco import COCO
from RCNN.models.models import classifier
from RCNN.utils.computations import selectiveSearch

from RCNN.utils.globalParams import Global
from RCNN.utils.metrics import non_max_suppression, softmax
from RCNN.utils.util import draw_box_with_text

def detect(model_path, model_name):

    transform = A.Compose(
        [
            A.Resize(
                height=Global.IMAGE_SIZE[0], width=Global.IMAGE_SIZE[1], always_apply=True, p=1),
            A.Normalize(always_apply=True, p=1),
            ToTensorV2()
        ]
    )

    model = classifier(model_path, model_name)

    ss = selectiveSearch()

    val_annot_path = Global.DATA_DIR + "annotations/instances_val2017.json"
    val_coco = COCO(val_annot_path)
    val_samples = val_coco.getImgIds()

    test_img_id = random.sample(val_samples, 1)[0]

    img_metadata = val_coco.loadImgs([test_img_id])
    image_name = img_metadata[0]['file_name'].split(".")[0]
    test_img_path = os.path.join(Global.DATA_DIR, "val/", image_name + ".jpg")

    img = cv2.imread(test_img_path)

    annIds = val_coco.getAnnIds(imgIds=[test_img_id])
    anns = val_coco.loadAnns(annIds)
    bndboxes = []

    for data in anns:
        x1, y1, x2, y2 = int(data["bbox"][0]), int(data["bbox"][1]), int(
            data["bbox"][0] + data["bbox"][2]), int(data["bbox"][1] + data["bbox"][3])
        if (x2 - x1) > 16 and (y2 - y1) > 16:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), Global.BBOX_COLOR, Global.BBOX_THICKNESS)
            
    ss.loadAlgo(img)
    rects = ss.getAnchors()

    positive_list = {k: {"rects": [], "scores": []} for k in range(1, Global.NUM_CLASSES)}

    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin: ymax, xmin: xmax]

        transformed_proposal = transform(image=rect_img)
        proposal = transformed_proposal["image"]

        proposal = proposal.to(Global.TORCH_DEVICE)
        output = model(proposal.unsqueeze(0))

        if torch.argmax(output).item() != 0:
            output_list = list(output.cpu().numpy()[0])
            probs = softmax(output_list)

            for i in range(1, len(probs)):

                if probs[i] > Global.SVM_THRESHOLD:
                    positive_list[i]["rects"].append(rect)
                    positive_list[i]["scores"].append(probs[i])

    for class_id in range(1, Global.NUM_CLASSES):

        pos_rects = positive_list[class_id]["rects"]
        pos_scores = positive_list[class_id]["scores"]

        if len(pos_rects) > 0 and len(pos_scores) > 0:
            nms_rects, nms_scores = non_max_suppression(pos_rects, pos_scores)
            draw_box_with_text(img, nms_rects, nms_scores)
    
    cv2.imwrite("./Sample.png", img)

for _ in range(10):
    classifier_path = "RCNN/models/checkpoints/svm_classifier/epoch_1_val_acc_0.7257.pt"
    detect(classifier_path, "vgg16")