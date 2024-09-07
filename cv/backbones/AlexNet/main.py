import torch
import traceback

from cv.utils import Global
from cv.backbones import AlexNet
from cv.configs.config import setup_config
from cv.datasets import ClassificationDataset
from cv.phases.classification import Train
from cv.src.tensorboard import TensorboardWriter
from cv.utils.logging_utils import deleteOldLogs, prepare_log_dir, start_logger

import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from cv.utils.pytorch_utils import setup_gpu_devices

if __name__ == "__main__":
    try:

        cfg = setup_config()

        if cfg.ASYNC_TRAINING:
            prepare_log_dir(cfg)
            setup_gpu_devices(log=False)
            world_size = torch.cuda.device_count()
            mp.spawn(Train.async_train, args=(cfg.LOGGING.NAME, world_size), nprocs=world_size, join=True)

        else:
            Global.setConfiguration(cfg)
            start_logger()
            setup_gpu_devices()
            
            Global.LOGGER.info("Configurations and Logger have been initialized")

            Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
            model = AlexNet(num_classes=1000)
            Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

            Global.LOGGER.info(f"Instantiating Optimizer: {cfg.AlexNet.OPTIMIZER.NAME}")
            optimizer = getattr(torch.optim, cfg.AlexNet.OPTIMIZER.NAME)(model.parameters(), **cfg.AlexNet.OPTIMIZER.PARAMS)
            Global.LOGGER.info(f"Optimizer instantiated")

            train_dataset = ClassificationDataset("train", transforms=cfg.AlexNet.TRANSFORMS.TRAIN, debug=None)
            val_dataset = ClassificationDataset("val", transforms=cfg.AlexNet.TRANSFORMS.VAL, debug=None)

            Global.LOGGER.info(f"Intantiating data loaders for training and validation")
            data_loaders = {}

            data_loaders["train"] = DataLoader(
                dataset=train_dataset, collate_fn=train_dataset.collate_fn,
                **cfg.AlexNet.DATALOADER_TRAIN_PARAMS
            )
            data_loaders["val"] = DataLoader(
                dataset=val_dataset, collate_fn=val_dataset.collate_fn,
                **cfg.AlexNet.DATALOADER_VAL_PARAMS
            )

            Global.LOGGER.info(f"Data loaders instantiated")

            Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.AlexNet.LR_SCHEDULER.NAME}")
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.AlexNet.LR_SCHEDULER.NAME
                )(optimizer, **cfg.AlexNet.LR_SCHEDULER.PARAMS)
            Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            Global.LOGGER.info(f"Instantiating Loss Function: {cfg.AlexNet.LOSS.NAME}")
            loss_function = getattr(torch.nn, cfg.AlexNet.LOSS.NAME)(**cfg.AlexNet.LOSS.PARAMS)
            Global.LOGGER.info(f"Loss Function instantiated")

            Global.LOGGER.info(f"Initializing Tensorboard writer")

            tb_writer = TensorboardWriter()

            Global.LOGGER.info(f"Starting training on device: {device}")

            train = Train(
                model=model,
                optimizer=optimizer,
                data_loaders=data_loaders,
                loss_function=loss_function,
                lr_scheduler=lr_scheduler,
                tb_writer=tb_writer,
                lr_tb_write_per_epoch=True,
                profiling=cfg.PROFILING,
                **cfg.TRAIN.PARAMS
            )
            train.start()

            Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()