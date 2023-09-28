import logging

class Global:
    LOGGER: logging.Logger = None
    LOG_FILENAME: str = None
    CFG = None

    @classmethod
    def setConfiguration(cls, cfg):
        Global.CFG = cfg

    @classmethod
    def setLogFilename(cls, log_filename):
        Global.LOG_FILENAME = log_filename