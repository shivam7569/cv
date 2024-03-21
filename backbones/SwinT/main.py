import traceback

import torch
from backbones import SwinT
from configs.config import setup_config
from datasets.classification.dataset import ClassificationDataset
from src.gpu_devices import GPU_Support
from utils.global_params import Global
from phases.classification.train import Train
from src.tensorboard import TensorboardWriter
from utils.logging_utils import clean_logs, deleteOldLogs, prepare_log_dir, start_logger

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from utils.pytorch_utils import async_cleanup, setup_gpu_devices

if __name__ == "__main__":
    try:
        cfg = setup_config()

        if cfg.ASYNC_TRAINING:
            prepare_log_dir(cfg)
            setup_gpu_devices(log=False)
            world_size = torch.cuda.device_count()
            mp.spawn(Train.async_train, args=("SwinT", world_size), nprocs=world_size, join=True)
        else:
            Global.setConfiguration(cfg)
            start_logger()
            setup_gpu_devices()

            if Global.CFG.DEBUG is not None:
                if Global.CFG.PROFILING:
                    Global.LOGGER.info(f"Running in debug mode, with profiling")
                else:
                    Global.LOGGER.info(f"Running in debug mode")

            Global.LOGGER.info("Configurations and Logger have been initialized")

            Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
            model = SwinT(**Global.CFG.SwinT.PARAMS)
            if GPU_Support.support_gpu:
                last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"
                model.to(last_gpu_id)
            Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

            Global.LOGGER.info(f"Instantiating Optimizer: {cfg.SwinT.OPTIMIZER.NAME}")
            optimizer = getattr(torch.optim, cfg.SwinT.OPTIMIZER.NAME)(model.parameters(), **cfg.SwinT.OPTIMIZER.PARAMS)
            Global.LOGGER.info(f"Optimizer instantiated")

            train_dataset = ClassificationDataset("train", transforms=cfg.SwinT.TRANSFORMS.TRAIN, debug=Global.CFG.DEBUG)
            val_dataset = ClassificationDataset("val", transforms=cfg.SwinT.TRANSFORMS.VAL, debug=Global.CFG.DEBUG // 2)

            Global.LOGGER.info(f"Intantiating data loaders for training and validation")
            data_loaders = {}

            data_loaders["train"] = DataLoader(
                dataset=train_dataset, collate_fn=train_dataset.collate_fn,
                **cfg.SwinT.DATALOADER_TRAIN_PARAMS
            )
            data_loaders["val"] = DataLoader(
                dataset=val_dataset, collate_fn=val_dataset.collate_fn,
                **cfg.SwinT.DATALOADER_VAL_PARAMS
            )

            Global.LOGGER.info(f"Data loaders instantiated")

            Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg.SwinT.LR_SCHEDULER.NAME}")

            lr_scheduler = Train.get_lr_scheduler(
                scheduler_name=cfg.SwinT.LR_SCHEDULER.NAME,
                optimizer=optimizer,
                scheduler_params=cfg.SwinT.LR_SCHEDULER.PARAMS
            )

            Global.LOGGER.info(f"Learning Rate Scheduler instantiated")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            Global.LOGGER.info(f"Instantiating Loss Function: {cfg.SwinT.LOSS.NAME}")
            loss_function = getattr(torch.nn, cfg.SwinT.LOSS.NAME)(**cfg.SwinT.LOSS.PARAMS)
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
                gradient_clipping=Global.CFG.SwinT.UTILITIES.GRADIENT_CLIPPING,
                epochs=100 if Global.CFG.DEBUG is None else 10,
                profiling=Global.CFG.PROFILING
            )

            train.start()

            Global.LOGGER.info(f"\nTraining completed")
        
    except Exception as e:
        Global.LOGGER.error(f"{type(e).__name__}: {e} \n{traceback.format_exc()}")
    finally:
        if cfg.ASYNC_TRAINING:
            async_cleanup()
        deleteOldLogs()
        clean_logs()
