import logging
import os

import torch
from configs.config import CfgNode as CN

torch.backends.cudnn.benchmark = True

_C = CN()

_C.DATA = CN()
_C.DATA.COCO_TRAIN_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/coco/train2017/"
_C.DATA.COCO_VAL_IMAGES_DIR = "/media/drive6/hqh2kor/datasets/coco/val2017/"
_C.DATA.COCO_TRAIN_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/coco/annotations/instances_train2017.json"
_C.DATA.COCO_VAL_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/coco/annotations/instances_val2017.json"
_C.DATA.COCO_ID_TO_CORRECT_ID = "/media/drive6/hqh2kor/projects/cv/datasets/coco_files/coco_id_to_ID.json"
_C.DATA.COCO_ID_TO_NAME = "/media/drive6/hqh2kor/projects/cv/datasets/coco_files/coco_id_to_name.json"
_C.DATA.COCO_ID_TO_COLOR = "/media/drive6/hqh2kor/projects/cv/datasets/coco_files/color_codes.json"
_C.DATA.COCO_EXCLUDE_IDS = "/media/drive6/hqh2kor/projects/cv/datasets/coco_files/empty_masks_train.json"

_C.DATA.IMAGENET_CLASS_MAPPING = "/media/drive6/hqh2kor/datasets/imagenet/LOC_synset_mapping.txt"
_C.DATA.IMAGENET_TRAIN_IMAGES = "/media/drive6/hqh2kor/datasets/imagenet/Data/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_IMAGES = "/media/drive6/hqh2kor/datasets/imagenet/Data/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/imagenet/Annotations/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_ANNOTATIONS = "/media/drive6/hqh2kor/datasets/imagenet/Annotations/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_TXT = "/media/drive6/hqh2kor/projects/cv/datasets/classification/imagenet_txts/train.txt"
_C.DATA.IMAGENET_VAL_TXT = "/media/drive6/hqh2kor/projects/cv/datasets/classification/imagenet_txts/val.txt"
_C.DATA.IMAGENET_CLASS_VS_ID_TXT = "/media/drive6/hqh2kor/projects/cv/datasets/classification/imagenet_txts/class_vs_id.txt"
_C.DATA.IMAGENET_CLASS_VS_NAME_TXT = "/media/drive6/hqh2kor/projects/cv/datasets/classification/imagenet_txts/class_vs_name.txt"

_C.LOGGING = CN()
_C.LOGGING.LEVEL = logging.INFO
_C.LOGGING.NAME = "Process"
_C.LOGGING.PATH = "/media/drive6/hqh2kor/projects/cv/logs/"

_C.METRICS = CN()
_C.METRICS.NAME = "Process"
_C.METRICS.PATH = "/media/drive6/hqh2kor/projects/cv/metrics/"
_C.METRICS.COLUMNS = ["epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1_score"]
_C.METRICS.SEG_COLUMNS = ["epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1_score", "eval_iou_score", "eval_dice_score", "eval_kappa_score"]

_C.CHECKPOINT = CN()
_C.CHECKPOINT.TREE = False
_C.CHECKPOINT.PATH = "/media/drive6/hqh2kor/projects/cv/checkpoints/"

_C.TENSORBOARD = CN()
_C.TENSORBOARD.PATH = "/media/drive6/hqh2kor/projects/cv/tensorboard/"

_C.PROFILER = CN()
_C.PROFILER.PATH = "/media/drive6/hqh2kor/projects/cv/profiler/"

_C.PATHS = CN()
_C.PATHS.BACKBONES = "/media/drive6/hqh2kor/projects/cv/backbones/"

_C.DATA_MIXING = CN()
_C.DATA_MIXING.enabled = False
_C.DATA_MIXING.one_hot_targets = False

_C.REPEAT_AUGMENTATIONS = False
_C.REPEAT_AUGMENTATIONS_NUM_REPEATS = 3

_C.RESUME_TRAINING = False
_C.DEBUG = None
_C.PROFILING = False
_C.EVALUATION_STEPS = 1

_C.SAVE_FIRST_SAMPLE = True
_C.WRITE_TENSORBOARD_GRAPH = True

os.environ["TORCH_HOME"] = "/media/drive6/hqh2kor/projects/cv/checkpoints/pytorch_hub/"