import os
import argparse
from cv.configs.config import setup_config
from cv.phases.classification import Train
from cv.utils.logging_utils import prepare_log_dir
from cv.utils.pytorch_utils import setup_gpu_devices

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Model processed"
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="model to use"
    )

    parser.add_argument(
        "--gpu-devices",
        required=False,
        help="gpu devices to use"
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    cfg = setup_config(args)
    prepare_log_dir(cfg, torchrun=True)
    world_size = setup_gpu_devices(args, log=False)
    Train.async_train(args=args, world_size=world_size, torchrun=True)