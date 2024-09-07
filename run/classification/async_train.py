import torch
import traceback
import torch.multiprocessing as mp

from cv.utils import Global
from cv.configs.config import setup_config
from cv.phases.classification import Train
from cv.utils.pytorch_utils import setup_gpu_devices
from cv.utils.logging_utils import deleteOldLogs, prepare_log_dir

if __name__ == "__main__":
    try:

        cfg = setup_config()

        if cfg.ASYNC_TRAINING:
            prepare_log_dir(cfg)
            setup_gpu_devices(log=False)
            world_size = torch.cuda.device_count()
            mp.spawn(Train.async_train, args=(cfg.LOGGING.NAME, world_size), nprocs=world_size, join=True)

        else:
            Global.LOGGER.warn("So you have decided to venture into the dark!!")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()