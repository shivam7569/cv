import logging
from configs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.COCO_TRAIN_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/datasets/train2017/"
_C.DATA.COCO_VAL_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/datasets/val2017/"
_C.DATA.COCO_TRAIN_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/datasets/annotations/instances_train2017.json"
_C.DATA.COCO_VAL_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/datasets/annotations/instances_val2017.json"

_C.DATA.IMAGENET_CLASS_MAPPING = "/media/drive6/hqh2kor/datasets/imagenet/LOC_synset_mapping.txt"
_C.DATA.IMAGENET_TRAIN_IMAGES = "/media/drive6/hqh2kor/datasets/imagenet/Data/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_IMAGES = "/media/drive6/hqh2kor/datasets/imagenet/Data/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/imagenet/Annotations/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/imagenet/Annotations/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_TXT = "/media/drive6/hqh2kor/projects/Computer_Vision/datasets/classification/imagenet_txts/train.txt"
_C.DATA.IMAGENET_VAL_TXT = "/media/drive6/hqh2kor/projects/Computer_Vision/datasets/classification/imagenet_txts/val.txt"
_C.DATA.IMAGENET_CLASS_VS_ID_TXT = "/media/drive6/hqh2kor/projects/Computer_Vision/datasets/classification/imagenet_txts/class_vs_id.txt"

_C.LOGGING = CN()
_C.LOGGING.LEVEL = logging.INFO
_C.LOGGING.NAME = "Process"
_C.LOGGING.PATH = "/media/drive6/hqh2kor/projects/Computer_Vision/logs/"