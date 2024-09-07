import os
import torch

from cv.utils import Global


class GPU_Support:

    support_gpu = 0

    @staticmethod
    def set_gpu_devices(gpu_devices=-1, log=True):
        if log: Global.LOGGER.info(f"Setting up GPU device(s)")
        
        if not isinstance(gpu_devices, (int, str)):
            raise TypeError(f"Invalid type for gpu_devices: {type(gpu_devices)}")
        
        if isinstance(gpu_devices, int) and gpu_devices != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices)
        elif isinstance(gpu_devices, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        elif gpu_devices == -1:
            if log: Global.LOGGER.info(f"No GPU devices specified, training on CPU")
        else:
            raise ValueError(f"Incompatible entry for gpu devices")
        
        try:
            _ = os.environ["CUDA_VISIBLE_DEVICES"]
            num_gpus = torch.cuda.device_count()
        except:
            num_gpus = 0

        GPU_Support.support_gpu = num_gpus