import traceback

import torch
from configs.config import setup_config
from utils.global_params import Global
from phases.segmentation.train import Train
from utils.logging_utils import deleteOldLogs, prepare_log_dir
import torch.multiprocessing as mp

from utils.pytorch_utils import setup_gpu_devices

if __name__ == "__main__":
    try:
        cfg = setup_config()

        if cfg.ASYNC_TRAINING:
            prepare_log_dir(cfg)
            setup_gpu_devices(log=False)
            world_size = torch.cuda.device_count()
            mp.spawn(Train.async_train, args=(cfg.LOGGING.NAME, world_size), nprocs=world_size, join=True)
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()