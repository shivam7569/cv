import logging

from cv.utils.imagenet_utils import ImagenetData

class MetaClass(type):
    def __new__(cls, name, bases, dct):
        new_class = super(MetaClass, cls).__new__(cls, name, bases, dct)
        new_class.setImagenetIdVsName()
        return new_class

class Global(metaclass=MetaClass):

    LOGGER: logging.Logger = None
    LOG_FILENAME: str = None
    CFG = None
    METRIC_CSV = None
    METRICS = {
        "epoch": None,
        "train_loss": None,
        "eval_loss": None,
        "eval_accuracy": None,
        "eval_precision": None,
        "eval_recall": None,
        "eval_f1_score": None
    }
    IMAGENET_ID_VS_NAME = None
    RUNTIME_PARAMS = {}

    @classmethod
    def setImagenetIdVsName(cls):
        cls.IMAGENET_ID_VS_NAME = ImagenetData.getIdVsName()

    @classmethod
    def setConfiguration(cls, cfg):
        Global.CFG = cfg

    @classmethod
    def setLogFilename(cls, log_filename):
        Global.LOG_FILENAME = log_filename

    @classmethod
    def resetEpochMetrics(cls):
        Global.METRICS = {
            "epoch": None,
            "train_loss": None,
            "eval_loss": None,
            "eval_accuracy": None,
            "eval_precision": None,
            "eval_recall": None,
            "eval_f1_score": None
        }

    @classmethod
    def addRuntimeParam(cls, key, value):
        if key not in Global.RUNTIME_PARAMS.keys():
            Global.RUNTIME_PARAMS[key] = value

    @classmethod
    def getRuntimeParam(cls, key):
        return Global.RUNTIME_PARAMS[key]