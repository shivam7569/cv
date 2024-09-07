import traceback

import torch
from backbones import MobileNet
from configs.config import setup_config
from datasets.classification.dataset import ClassificationDataset
from utils.global_params import Global
from phases.classification.train import Train
from src.tensorboard import TensorboardWriter
from utils.logging_utils import deleteOldLogs, start_logger

from torch.utils.data import DataLoader

from utils.pytorch_utils import setup_gpu_devices

if __name__ == "__main__":
    try:
        cfg = setup_config()
        Global.setConfiguration(cfg)
        start_logger()
        setup_gpu_devices()

        Global.LOGGER.info("Configurations and Logger have been initialized")

        Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
        model = MobileNet(num_classes=1000, alpha=1)
        Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg.MobileNet.OPTIMIZER.NAME}")
        optimizer = getattr(torch.optim, cfg.MobileNet.OPTIMIZER.NAME)(model.parameters(), **cfg.MobileNet.OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        train_dataset = ClassificationDataset("train", transforms=cfg.MobileNet.TRANSFORMS.TRAIN, debug=None)
        val_dataset = ClassificationDataset("val", transforms=cfg.MobileNet.TRANSFORMS.VAL, debug=None)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}

        data_loaders["train"] = DataLoader(
            dataset=train_dataset,
            **cfg.MobileNet.DATALOADER_TRAIN_PARAMS
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset,
            **cfg.MobileNet.DATALOADER_VAL_PARAMS
        )

        Global.LOGGER.info(f"Data loaders instantiated")

        Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.MobileNet.LR_SCHEDULER.NAME}")

        if cfg.MobileNet.LR_SCHEDULER.NAME == "LambdaLR":
            lr_lambda = lambda epoch: 0.96 * epoch
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.MobileNet.LR_SCHEDULER.NAME
                )(optimizer, lr_lambda=lr_lambda, **cfg.MobileNet.LR_SCHEDULER.PARAMS)
        elif cfg.MobileNet.LR_SCHEDULER.NAME == "MultiplicativeLR":
            lr_lambda = lambda epoch: cfg.MobileNet.LR_SCHEDULER.FACTOR
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.MobileNet.LR_SCHEDULER.NAME
            )(optimizer, lr_lambda=lr_lambda, **cfg.MobileNet.LR_SCHEDULER.PARAMS)
        else: 
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.MobileNet.LR_SCHEDULER.NAME
                )(optimizer, **cfg.MobileNet.LR_SCHEDULER.PARAMS)
        Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        Global.LOGGER.info(f"Instantiating Loss Function: {cfg.MobileNet.LOSS.NAME}")
        loss_function = getattr(torch.nn, cfg.MobileNet.LOSS.NAME)(**cfg.MobileNet.LOSS.PARAMS)
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
            epochs=500
        )
        train.start()

        Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()