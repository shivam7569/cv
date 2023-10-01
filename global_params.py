import logging

import pandas as pd

class Global:
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
