import traceback

import torch
from .model import DenseNet
from cv.configs.config import setup_config
from cv.datasets.classification.dataset import ClassificationDataset
from cv.utils.global_params import Global
from cv.phases.classification.train import Train
from cv.src.tensorboard import TensorboardWriter
from cv.utils.logging_utils import deleteOldLogs, start_logger

from torch.utils.data import DataLoader

from cv.utils.pytorch_utils import setup_gpu_devices

if __name__ == "__main__":
    try:
        cfg = setup_config()
        Global.setConfiguration(cfg)
        start_logger()
        setup_gpu_devices()

        Global.LOGGER.info("Configurations and Logger have been initialized")

        Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
        model = DenseNet(num_classes=1000)
        Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg.DenseNet.OPTIMIZER.NAME}")
        optimizer = getattr(torch.optim, cfg.DenseNet.OPTIMIZER.NAME)(model.parameters(), **cfg.DenseNet.OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        train_dataset = ClassificationDataset("train", transforms=cfg.DenseNet.TRANSFORMS.TRAIN, debug=None)
        val_dataset = ClassificationDataset("val", transforms=cfg.DenseNet.TRANSFORMS.VAL, debug=None)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}

        data_loaders["train"] = DataLoader(
            dataset=train_dataset,
            **cfg.DenseNet.DATALOADER_TRAIN_PARAMS
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset,
            **cfg.DenseNet.DATALOADER_VAL_PARAMS
        )

        Global.LOGGER.info(f"Data loaders instantiated")

        Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.DenseNet.LR_SCHEDULER.NAME}")

        if cfg.DenseNet.LR_SCHEDULER.NAME == "LambdaLR":
            lr_lambda = lambda epoch: 0.96 * epoch
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.DenseNet.LR_SCHEDULER.NAME
                )(optimizer, lr_lambda=lr_lambda, **cfg.DenseNet.LR_SCHEDULER.PARAMS)
        else: 
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, cfg.DenseNet.LR_SCHEDULER.NAME
                )(optimizer, **cfg.DenseNet.LR_SCHEDULER.PARAMS)
        Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        Global.LOGGER.info(f"Instantiating Loss Function: {cfg.DenseNet.LOSS.NAME}")
        loss_function = getattr(torch.nn, cfg.DenseNet.LOSS.NAME)(**cfg.DenseNet.LOSS.PARAMS)
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
            epochs=1000
        )
        train.start()

        Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        deleteOldLogs()