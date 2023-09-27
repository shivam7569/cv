import traceback
from LeNet.config import setup_config
from coco.detection.dataset import DetectionDataset
from coco.segmentation.dataset import SegmentationDataset
from global_params import Global
from utils.logging_utils import deleteOldLogs, start_logger


if __name__ == "__main__":
    try:
        cfg = setup_config()
        Global.setConfiguration(cfg)
        start_logger()
        
        Global.LOGGER.info("All set!!")
        
        dt = SegmentationDataset("val")
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()