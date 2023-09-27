import logging
from configs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.TRAIN_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/coco/train2017/"
_C.DATA.VAL_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/coco/val2017/"
_C.DATA.TRAIN_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/coco/annotations/instances_train2017.json"
_C.DATA.VAL_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/coco/annotations/instances_val2017.json"

_C.LOGGING = CN()
_C.LOGGING.LEVEL = logging.INFO
_C.LOGGING.NAME = "Process"
_C.LOGGING.PATH = "/media/drive6/hqh2kor/projects/Computer_Vision/logs/"