import traceback

import torch
from backbones import SENet
from configs.config import setup_config
from datasets.classification.dataset import ClassificationDataset
from utils.global_params import Global
from phases.classification.train import Train
from src.tensorboard import TensorboardWriter
from utils.logging_utils import deleteOldLogs, prepare_log_dir, start_logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from utils.pytorch_utils import setup_gpu_devices

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
            model = SENet(num_classes=1000, reduction_ratio=16)
            Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

            Global.LOGGER.info(f"Instantiating Optimizer: {cfg.SENet.OPTIMIZER.NAME}")
            optimizer = getattr(torch.optim, cfg.SENet.OPTIMIZER.NAME)(model.parameters(), **cfg.SENet.OPTIMIZER.PARAMS)
            Global.LOGGER.info(f"Optimizer instantiated")

            train_dataset = ClassificationDataset("train", transforms=cfg.SENet.TRANSFORMS.TRAIN, debug=None)
            val_dataset = ClassificationDataset("val", transforms=cfg.SENet.TRANSFORMS.VAL, debug=None)

            Global.LOGGER.info(f"Intantiating data loaders for training and validation")
            data_loaders = {}

            data_loaders["train"] = DataLoader(
                dataset=train_dataset,
                **cfg.SENet.DATALOADER_TRAIN_PARAMS
            )
            data_loaders["val"] = DataLoader(
                dataset=val_dataset,
                **cfg.SENet.DATALOADER_VAL_PARAMS
            )

            Global.LOGGER.info(f"Data loaders instantiated")

            Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.SENet.LR_SCHEDULER.NAME}")

            if cfg.SENet.LR_SCHEDULER.NAME == "LambdaLR":
                lr_lambda = lambda epoch: 0.96 * epoch
                lr_scheduler = getattr(
                    torch.optim.lr_scheduler, cfg.SENet.LR_SCHEDULER.NAME
                    )(optimizer, lr_lambda=lr_lambda, **cfg.SENet.LR_SCHEDULER.PARAMS)
            elif cfg.SENet.LR_SCHEDULER.NAME == "MultiplicativeLR":
                lr_lambda = lambda epoch: cfg.SENet.LR_SCHEDULER.FACTOR
                lr_scheduler = getattr(
                    torch.optim.lr_scheduler, cfg.SENet.LR_SCHEDULER.NAME
                )(optimizer, lr_lambda=lr_lambda, **cfg.SENet.LR_SCHEDULER.PARAMS)
            else: 
                lr_scheduler = getattr(
                    torch.optim.lr_scheduler, cfg.SENet.LR_SCHEDULER.NAME
                    )(optimizer, **cfg.SENet.LR_SCHEDULER.PARAMS)
            Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            Global.LOGGER.info(f"Instantiating Loss Function: {cfg.SENet.LOSS.NAME}")
            loss_function = getattr(torch.nn, cfg.SENet.LOSS.NAME)(**cfg.SENet.LOSS.PARAMS)
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
                epochs=5000
            )
            train.start()

            Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()