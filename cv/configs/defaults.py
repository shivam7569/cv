import logging
import os

import torch
from cv.configs.config import CfgNode as CN

torch.backends.cudnn.benchmark = True

_C = CN()

project_dir = os.getenv("PROJECT_DIR", default=os.path.abspath(os.path.join(__file__, "../../..")))

coco_data_dir = os.getenv("COCO_DIR", default="/media/drive6/hqh2kor/datasets/coco")
imagenet_data_dir = os.getenv("IMAGENET_DIR", default="/media/drive6/hqh2kor/datasets/imagenet")
mnist_data_dir = os.getenv("MNIST_DIR", default="/media/drive6/hqh2kor/datasets/mnist")

_C.DATA = CN()
_C.DATA.COCO_TRAIN_IMAGES_DIR = f"{coco_data_dir}/train2017/"
_C.DATA.COCO_VAL_IMAGES_DIR = f"{coco_data_dir}/val2017/"
_C.DATA.COCO_TRAIN_ANNOTATIONS = f"{coco_data_dir}/annotations/instances_train2017.json"
_C.DATA.COCO_VAL_ANNOTATIONS = f"{coco_data_dir}/annotations/instances_val2017.json"
_C.DATA.COCO_ID_TO_CORRECT_ID = f"{project_dir}/cv/datasets/coco_files/coco_id_to_ID.json"
_C.DATA.COCO_ID_TO_NAME = f"{project_dir}/cv/datasets/coco_files/coco_id_to_name.json"
_C.DATA.COCO_ID_TO_COLOR = f"{project_dir}/cv/datasets/coco_files/color_codes.json"
_C.DATA.COCO_EXCLUDE_IDS = f"{project_dir}/cv/datasets/coco_files/empty_masks_train.json"

_C.DATA.IMAGENET_CLASS_MAPPING = f"{imagenet_data_dir}/LOC_synset_mapping.txt"
_C.DATA.IMAGENET_TRAIN_IMAGES = f"{imagenet_data_dir}/Data/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_IMAGES = f"{imagenet_data_dir}/Data/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_ANNOTATIONS = f"{imagenet_data_dir}/Annotations/CLS-LOC/train/"
_C.DATA.IMAGENET_VAL_ANNOTATIONS = f"{imagenet_data_dir}/Annotations/CLS-LOC/val/"
_C.DATA.IMAGENET_TRAIN_TXT = f"{project_dir}/cv/datasets/classification/imagenet_txts/train.txt"
_C.DATA.IMAGENET_VAL_TXT = f"{project_dir}/cv/datasets/classification/imagenet_txts/val.txt"
_C.DATA.IMAGENET_CLASS_VS_ID_TXT = f"{project_dir}/cv/datasets/classification/imagenet_txts/class_vs_id.txt"
_C.DATA.IMAGENET_CLASS_VS_NAME_TXT = f"{project_dir}/cv/datasets/classification/imagenet_txts/class_vs_name.txt"

_C.DATA.MNIST_TRAIN_CSV = f"{mnist_data_dir}/mnist_train.csv"
_C.DATA.MNIST_TEST_CSV = f"{mnist_data_dir}/mnist_test.csv"
_C.DATA.MNIST_TRAIN_TXT = f"{project_dir}/cv/datasets/mnist_txts/train.txt"
_C.DATA.MNIST_TEST_TXT = f"{project_dir}/cv/datasets/mnist_txts/test.txt"

_C.LOGGING = CN()
_C.LOGGING.LEVEL = logging.INFO
_C.LOGGING.NAME = "Process"
_C.LOGGING.PATH = f"{project_dir}/logs/"

_C.METRICS = CN()
_C.METRICS.NAME = "Process"
_C.METRICS.PATH = f"{project_dir}/metrics/"
_C.METRICS.COLUMNS = ["epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1_score"]
_C.METRICS.SEG_COLUMNS = ["epoch", "train_loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1_score", "eval_iou_score", "eval_dice_score", "eval_kappa_score"]

_C.CHECKPOINT = CN()
_C.CHECKPOINT.TREE = False
_C.CHECKPOINT.PATH = f"{project_dir}/checkpoints/"

_C.TENSORBOARD = CN()
_C.TENSORBOARD.PATH = f"{project_dir}/tensorboard/"

_C.PROFILER = CN()
_C.PROFILER.PATH = f"{project_dir}/profiler/"
_C.PROFILER.STEPS = CN()
_C.PROFILER.STEPS.wait = 125
_C.PROFILER.STEPS.warmup = 125
_C.PROFILER.STEPS.active = 250
_C.PROFILER.STEPS.repeat = 5

_C.PATHS = CN()
_C.PATHS.BACKBONES = f"{project_dir}/cv/backbones/"

_C.FONT_PATH = f"{project_dir}/cv/utils/_files/Aileron-Black.otf"
_C.BPE_VOCAB_PATH = f"{project_dir}/cv/utils/_files/bpe_simple_vocab_16e6.txt.gz"

_C.DATA_MIXING = CN()
_C.DATA_MIXING.enabled = False
_C.DATA_MIXING.one_hot_targets = False

_C.REPEAT_AUGMENTATIONS = False
_C.REPEAT_AUGMENTATIONS_NUM_REPEATS = 3

_C.RESUME_TRAINING = False
_C.DEBUG = None
_C.PROFILING = False
_C.EVALUATION_STEPS = 1
_C.USE_SYNC_BN = True
_C.SCHEDULE_FREE_TRAINING = False

_C.SAVE_FIRST_SAMPLE = True
_C.WRITE_TENSORBOARD_GRAPH = True

_C.COCO_MEAN = [0.470, 0.447, 0.408]
_C.COCO_STD = [0.278, 0.274, 0.289]

os.environ["TORCH_HOME"] = f"{project_dir}/checkpoints/pytorch_hub/"