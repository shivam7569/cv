import traceback

import torch
from backbones.LeNet.model import LeNet
from configs.config import setup_config
from datasets.classification.dataset import ClassificationDataset
from utils.global_params import Global
from phases.classification.train import Train
from src.tensorboard import TensorboardWriter
from utils.logging_utils import deleteOldLogs, start_logger

from torch.utils.data import DataLoader

if __name__ == "__main__":
    try:
        cfg = setup_config()
        Global.setConfiguration(cfg)
        start_logger()
        Global.LOGGER.info("Configurations and Logger have been initialized")

        Global.LOGGER.info("Instantiating LeNet Architecture for classification on 1000 classes")
        model = LeNet(num_classes=1000)
        Global.LOGGER.info("LeNet Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg.LeNet.OPTIMIZER.NAME}")
        optimizer = getattr(torch.optim, cfg.LeNet.OPTIMIZER.NAME)(model.parameters(), **cfg.LeNet.OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        train_dataset = ClassificationDataset("train", transforms=cfg.LeNet.TRANSFORMS.TRAIN)
        val_dataset = ClassificationDataset("val", transforms=cfg.LeNet.TRANSFORMS.VAL)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}

        data_loaders["train"] = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        Global.LOGGER.info(f"Data loaders instantiated")

        lr_scheduler = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        Global.LOGGER.info(f"Instantiating Loss Function: {cfg.LeNet.LOSS.NAME}")
        loss_function = getattr(torch.nn, cfg.LeNet.LOSS.NAME)(**cfg.LeNet.LOSS.PARAMS)
        Global.LOGGER.info(f"Loss Function instantiated")

        Global.LOGGER.info(f"Initializing Tensorboard writer")

        tb_writer = TensorboardWriter()

        Global.LOGGER.info(f"Starting training on device: {device}")

        train = Train(
            model=model,
            optimizer=optimizer,
            data_loaders=data_loaders,
            device=device,
            loss_function=loss_function,
            lr_scheduler=lr_scheduler,
            tb_writer=tb_writer,
            epochs=100
        )
        train.start()

        Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()