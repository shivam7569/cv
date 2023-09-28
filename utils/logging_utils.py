import os
import logging
from datetime import datetime
import traceback
from global_params import Global
from utils.os_utils import check_dir

def initialize_logger():

    log_filename = f"{Global.CFG.LOGGING.NAME}-{getCurrentDateTime()}.log"

    Global.setLogFilename(log_filename)

    log_filepath = os.path.join(Global.CFG.LOGGING.PATH, Global.CFG.LOGGING.NAME)

    check_dir(Global.CFG.LOGGING.PATH, create=True)
    check_dir(log_filepath, create=True)

    log_filepath = os.path.join(log_filepath, log_filename)

    logger = logging.getLogger(name=Global.CFG.LOGGING.NAME)
    logging.basicConfig(
        level=Global.CFG.LOGGING.LEVEL
    )

    return logger, log_filepath

def getCurrentDateTime() -> str:
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    return dt_string

class LogLevelFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= Global.CFG.LOGGING.LEVEL
    
class LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = f"{record.levelname} => {record.msg}"

        return message
    
class StackTraceLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stack_lines = traceback.format_stack()
        required_stack_lines = filter(
            lambda line: ("logging/__init__.py" not in line)
            and ("debugpy" not in line) and (("runpy" not in line)),
            stack_lines[:-1]
        )
        message = record.levelname + " => " + record.msg + f"\nOrigin: \n" + "".join(required_stack_lines)
        message = message.rstrip('\n')

        return message

def customLogHandlers():

    LOGGER, logg_filepath = initialize_logger()
    Global.LOGGER = LOGGER

    LOGGER.handlers = []

    regular_handler = logging.FileHandler(logg_filepath, mode='a', encoding="utf-8")
    regular_handler.setLevel(Global.CFG.LOGGING.LEVEL)
    regular_handler.addFilter(LogLevelFilter())
    regular_handler.setFormatter(LogFormatter())

    LOGGER.addHandler(regular_handler)

    custom_handler = logging.FileHandler(logg_filepath, mode='a', encoding="utf-8")
    custom_handler.setLevel(Global.CFG.LOGGING.LEVEL + 20)
    custom_handler.setFormatter(StackTraceLogFormatter())

    LOGGER.addHandler(custom_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(Global.CFG.LOGGING.LEVEL)
    LOGGER.addHandler(stream_handler)

    LOGGER.propagate = False

def deleteOldLogs() -> None:

    log_dir = os.path.join(Global.CFG.LOGGING.PATH, Global.CFG.LOGGING.NAME)
    log_files = sorted(os.listdir(log_dir))[:-1]
    if len(log_files) > 0:
        for log in log_files:
            os.remove(os.path.join(log_dir, log))

def start_logger():
    customLogHandlers()

    Global.LOGGER.info(f"Logger started, log file stored as: {Global.LOG_FILENAME}")