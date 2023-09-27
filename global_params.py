import logging

class Global:
    LOGGER: logging.Logger = None
    CFG = None

    @classmethod
    def setConfiguration(cls, cfg):
        Global.CFG = cfg