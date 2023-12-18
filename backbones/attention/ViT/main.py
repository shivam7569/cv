import traceback

import torch
from backbones import ViT
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

        if Global.CFG.DEBUG is not None:
            Global.LOGGER.info(f"Running in debug mode, with profiling")

        Global.LOGGER.info("Configurations and Logger have been initialized")

        Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
        model = ViT(**Global.CFG.ViT.PARAMS)
        Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg.ViT.OPTIMIZER.NAME}")
        optimizer = getattr(torch.optim, cfg.ViT.OPTIMIZER.NAME)(model.parameters(), **cfg.ViT.OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        train_dataset = ClassificationDataset("train", transforms=cfg.ViT.TRANSFORMS.TRAIN, debug=Global.CFG.DEBUG)
        val_dataset = ClassificationDataset("val", transforms=cfg.ViT.TRANSFORMS.VAL, debug=Global.CFG.DEBUG // 2)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}

        data_loaders["train"] = DataLoader(
            dataset=train_dataset, collate_fn=train_dataset.collate_fn,
            **cfg.ViT.DATALOADER_TRAIN_PARAMS
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset, collate_fn=val_dataset.collate_fn,
            **cfg.ViT.DATALOADER_VAL_PARAMS
        )

        Global.LOGGER.info(f"Data loaders instantiated")

        Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.ViT.LR_SCHEDULER.NAME}")

        lr_scheduler = Train.get_lr_scheduler(
            scheduler_name=cfg.ViT.LR_SCHEDULER.NAME,
            optimizer=optimizer,
            scheduler_params=cfg.ViT.LR_SCHEDULER.PARAMS
        )

        Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        Global.LOGGER.info(f"Instantiating Loss Function: {cfg.ViT.LOSS.NAME}")
        loss_function = getattr(torch.nn, cfg.ViT.LOSS.NAME)(**cfg.ViT.LOSS.PARAMS)
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
            gradient_clipping=Global.CFG.ViT.UTILITIES.GRADIENT_CLIPPING,
            epochs=100 if Global.CFG.DEBUG is None else 10,
            profiling=Global.CFG.PROFILING
        )

        train.start()

        Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()
