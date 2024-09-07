import traceback

import torch
from backbones import Inceptionv2
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
        model = Inceptionv2(num_classes=1000)
        Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg.Inceptionv2.OPTIMIZER.NAME}")
        optimizer = getattr(torch.optim, cfg.Inceptionv2.OPTIMIZER.NAME)(model.parameters(), **cfg.Inceptionv2.OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        train_dataset = ClassificationDataset("train", transforms=cfg.Inceptionv2.TRANSFORMS.TRAIN, debug=None)
        val_dataset = ClassificationDataset("val", transforms=cfg.Inceptionv2.TRANSFORMS.VAL, debug=None)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}

        data_loaders["train"] = DataLoader(
            dataset=train_dataset,
            **cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset,
            **cfg.Inceptionv2.DATALOADER_VAL_PARAMS
        )

        Global.LOGGER.info(f"Data loaders instantiated")

        Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.Inceptionv2.LR_SCHEDULER.NAME}")

        if cfg.Inceptionv2.LR_SCHEDULER.NAME == "LambdaLR":
            lr_lambda = lambda epoch: 0.96 * epoch
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.Inceptionv2.LR_SCHEDULER.NAME
                )(optimizer, lr_lambda=lr_lambda, **cfg.Inceptionv2.LR_SCHEDULER.PARAMS)
        else: 
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.Inceptionv2.LR_SCHEDULER.NAME
                )(optimizer, **cfg.Inceptionv2.LR_SCHEDULER.PARAMS)
        Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        Global.LOGGER.info(f"Instantiating Loss Function: {cfg.Inceptionv2.LOSS.NAME}")
        loss_function = getattr(torch.nn, cfg.Inceptionv2.LOSS.NAME)(**cfg.Inceptionv2.LOSS.PARAMS)
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
            epochs=1000,
            phase_dependent=True,
            custom_loss="inceptionv2_loss"
        )
        train.start()

        Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()