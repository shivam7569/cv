import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from cv.utils import Global
from cv.utils import MetaWrapper
import cv.contrastive as contrastive
from cv.utils.os_utils import check_dir
import cv.src.losses as custom_loss_fns
from cv.src import custom_lrs, optimizers
from cv.src.checkpoints import Checkpoint
from cv.phases.contrastive import Eval
from cv.src.gpu_devices import GPU_Support
from cv.configs.config import setup_config
from cv.datasets import ContrastiveDataset
from cv.utils.logging_utils import start_logger
from cv.src.tensorboard import TensorboardWriter
from cv.utils.training import EMA, RepeatAugSampler
from cv.utils.pytorch_utils import async_parallel_setup


class Train(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Training Class for all the contrastive training methodologies"
    
    _ACTIVITY_MAPS = {
        "CPU": torch.profiler.ProfilerActivity.CPU,
        "CUDA": torch.profiler.ProfilerActivity.CUDA
    }

    def __init__(
            self,
            model,
            optimizer,
            data_loaders,
            loss_function,
            lr_scheduler=None,
            lr_scheduler_step=None,
            tb_writer=None,
            lr_tb_write_per_epoch=True,
            epochs=100,
            phase_dependent=False,
            custom_loss=None,
            gradient_clipping=None,
            profiling=None,
            resume_epoch=0,
            async_parallel=False,
            async_parallel_rank=0,
            gradient_accumulation=False,
            gradient_accumulation_batch_size=None,
            exponential_moving_average=None,
            updateStochasticDepthRate=None
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.loss_function = loss_function
        self.epochs = epochs
        self.tb_writer = tb_writer
        self.lr_tb_write_per_epoch = lr_tb_write_per_epoch
        self.phase_dependent = phase_dependent
        self.custom_loss = custom_loss
        self.gradient_clipping = gradient_clipping
        self.resume_epoch = resume_epoch
        self.updateStochasticDepthRate = updateStochasticDepthRate

        self.train_loader = data_loaders["train"]
        self.val_loader = data_loaders["val"]

        self.profiling = profiling
        if profiling:
            Global.LOGGER.info(f"Profiler is being used for bottleneck analysis")
            if profiling: self._setup_profiler()

        self.async_parallel = async_parallel
        self.async_parallel_rank = async_parallel_rank

        self.gradient_accumulation = gradient_accumulation
        if gradient_accumulation:
            self.gradient_accumulation_batch_size = gradient_accumulation_batch_size
            assert gradient_accumulation_batch_size is not None and isinstance(gradient_accumulation_batch_size, int)
            self.accumulation_iter = gradient_accumulation_batch_size // self.train_loader.batch_size

        self.exponential_moving_average = exponential_moving_average
        if exponential_moving_average is not None:
            self.ema_model = EMA(
                model=self.model.module, 
                **exponential_moving_average) \
                    if async_parallel else EMA(
                        model=self.model,
                        **exponential_moving_average)

        if async_parallel:
            Global.LOGGER.info(f"Training asynchronously with parallel model distribution")

        if not async_parallel_rank:
            self.evaluation = Eval(
                model=model if exponential_moving_average is None else self.ema_model,
                data_loader=data_loaders["val"],
                loss_function=loss_function,
                tb_writer=tb_writer,
                async_parallel=async_parallel,
                async_parallel_rank=async_parallel_rank
            )
            self.checkpointer = Checkpoint(
                model if not async_parallel else model.module, optimizer, lr_scheduler
            )
        
        self.graph_written = False
        self.sample_batch_log = False

        self.log_train_setting()

    def start(self):
        if not self.async_parallel_rank: f1_score = np.NINF
        for epoch in range(self.resume_epoch, self.epochs):

            if self.async_parallel:
                self.train_loader.sampler.set_epoch(epoch)

            if self.tb_writer is not None: self.tb_writer.setWriter("train")

            if self.profiling:
                self.profiler.start()
                epoch_loss = self._run_profiling(epoch=epoch)
            else:
                epoch_loss = self.train_for_one_epoch(epoch=epoch)

            if self.exponential_moving_average is not None:
                ema_beta_val =self._exponential_moving_average(last_epoch=epoch==self.epochs-1)
                if self.tb_writer is not None:
                    self.tb_writer.write("scaler")(scalar_name="EMA beta", scalar_value=ema_beta_val, step=epoch)

            if self.tb_writer is not None: self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=epoch_loss, step=epoch)

            if not self.async_parallel_rank:
                Global.METRICS["epoch"] = epoch
                Global.METRICS["train_loss"] = epoch_loss
                
                if Global.CFG.SCHEDULE_FREE_TRAINING: self.optimizer.eval()
                self.evaluation.start(epoch=epoch)

                if self.evaluation.epoch_metrics["f1_score"] > f1_score:
                    f1_score = self.evaluation.epoch_metrics["f1_score"]
                    self._mark_checkpoint(epoch=epoch, f1_score=f1_score, epoch_chkpt=False)
                if Global.CFG.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS:
                    self._mark_checkpoint(epoch=epoch, f1_score=None, epoch_chkpt=True)

                Global.resetEpochMetrics()

            if self.lr_scheduler is not None:
                self._lr_scheduling(epoch_based=True, epoch=epoch)

            if self.lr_tb_write_per_epoch:
                if self.tb_writer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.write("scaler")(scalar_name="Learning Rate", scalar_value=current_lr, step=epoch)

            if self.updateStochasticDepthRate is not None:
                if epoch % self.updateStochasticDepthRate[0]["step_epochs"] == 0 and epoch > 0:
                    self.model.module.updateStochasticDepthRate(self.updateStochasticDepthRate[0]["k"])

    def train_for_one_epoch(self, epoch, profiler=None):

        self.model.train()

        if Global.CFG.SCHEDULE_FREE_TRAINING: self.optimizer.train()

        num_iterations = len(self.train_loader)
        data_iterator = tqdm(self.train_loader, desc=f"Training: Epoch {epoch+1}", unit="batch") if not self.async_parallel_rank else self.train_loader
        
        if not self.async_parallel_rank: epoch_loss = 0
        for idx, batch in enumerate(data_iterator):

            img_batch, _ = batch

            if Global.CFG.SAVE_FIRST_SAMPLE:
                if (self.tb_writer is not None) and (not self.sample_batch_log) and (epoch == 0):
                    self.tb_writer.write("image")(image=ContrastiveDataset._vizualizeBatch(images=img_batch), epoch=epoch+1, tag="Training Sample")
                    self.sample_batch_log = True

            if not self.async_parallel:
                if GPU_Support.support_gpu:
                    last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"
            
                    img_batch = img_batch.to(last_gpu_id)
            else:
                img_batch = img_batch.to(self.async_parallel_rank, non_blocking=True)

            if (self.tb_writer is not None) and (not self.graph_written) and (Global.CFG.WRITE_TENSORBOARD_GRAPH):
                tb_model = self.model if not self.async_parallel else self.model.module
                self.tb_writer.write("graph")(model=tb_model, input_to_model=img_batch)
                self.graph_written = True

            if self.phase_dependent:
                output = self.model(img_batch, phase="training")
            else:
                output = self.model(img_batch)

            if self.phase_dependent:
                loss = self.loss_function(output, phase="training")
            else:
                loss = self.loss_function(output)

            if Global.CFG.REGULARIZATION.MODE in ["L1", "L2"]:
                loss = self._regularize(loss=loss)

            loss.backward()

            self._optimizer_step(batch_idx=idx, len_loader=num_iterations)

            if not self.async_parallel_rank: epoch_loss += loss.item()

            if self.lr_scheduler is not None:
                batch_iteration = idx/num_iterations if self.lr_scheduler.__class__.__name__ in ["CosineAnnealingWarmRestarts", "WarmUpLinearLRScheduler"] else idx
                self._lr_scheduling(epoch_based=False, epoch=epoch, batch_iteration=batch_iteration)

            if not self.async_parallel_rank: data_iterator.set_postfix(loss=loss.item(), refresh=True)
            
            if not self.lr_tb_write_per_epoch:
                if self.tb_writer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.write("scaler")(scalar_name="Learning Rate", scalar_value=current_lr, step=epoch * num_iterations + idx)

            if profiler is not None:
                profiler.step()
                if idx >= (sum([v for k, v in Global.CFG.PROFILER.STEPS.items() if k != "repeat"]) * Global.CFG.PROFILER.STEPS["repeat"]):
                    self.profiler.stop()
                    break

        if not self.async_parallel_rank:
            epoch_loss /= num_iterations
            Global.LOGGER.info(f"\nTraining loss for epoch {epoch+1}: {round(epoch_loss, 3)}")
            
        if not self.async_parallel_rank: return epoch_loss

    def _optimizer_step(self, batch_idx, len_loader):
        if self.gradient_accumulation:
            if (batch_idx % self.accumulation_iter == 0) or (batch_idx + 1 == len_loader):
                self._gradient_clipping()
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self._gradient_clipping()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _gradient_clipping(self):
        if self.gradient_clipping is not None:
            clipping_method, clipping_threshold = self.gradient_clipping
            if clipping_method == "val":
                torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), clip_value=clipping_threshold)
            if clipping_method == "norm":
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=clipping_threshold)

    def _mark_checkpoint(self, epoch, f1_score, epoch_chkpt):
        if not epoch_chkpt:
            if '/' in Global.CFG.CHECKPOINT.BASENAME:
                    basename = Global.CFG.CHECKPOINT.BASENAME.split("/")[-1]
            else:
                basename = Global.CFG.CHECKPOINT.BASENAME
            checkpoint_name = basename + f"_epoch_{epoch+1}_f1_{f1_score}.pth"
            Global.LOGGER.info(f"Saving metric checkpoint for epoch {epoch+1}")
            self.checkpointer.save(
                epoch=epoch,
                chkp_name=checkpoint_name,
                overwrite=True
            )
        else:
            self.checkpointer.save(epoch=epoch, chkp_name="epoch_checkpoint.pth", overwrite=False)

    def _exponential_moving_average(self, last_epoch):
        if self.exponential_moving_average is not None:
            current_beta_val = self.ema_model.update(last_epoch)

            return current_beta_val

    def _lr_scheduling(self, epoch_based, epoch=None, batch_iteration=None):
        if epoch_based:
            assert epoch is not None

            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(self.evaluation.epoch_metrics["eval_loss"])
            if self.lr_scheduler.__class__.__name__ in ["MultiStepLR", "StepLR"]:
                self.lr_scheduler.step()
            if self.lr_scheduler.__class__.__name__ == "ExponentialLR":
                if (epoch + 1) % self.lr_scheduler_step == 0:
                    self.lr_scheduler.step()
            if self.lr_scheduler.__class__.__name__ == "MultiplicativeLR":
                self.lr_scheduler.step()
            if self.lr_scheduler.__class__.__name__ == "WarmUpCosineLRScheduler":
                self.lr_scheduler.step()                
        else:
            if self.lr_scheduler.__class__.__name__ == "CyclicLR":
                self.lr_scheduler.step()
            if self.lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                self.lr_scheduler.step(epoch + batch_iteration)
            if self.lr_scheduler.__class__.__name__ == "WarmUpLinearLRScheduler":
                self.lr_scheduler.step(epoch + batch_iteration)

    def _regularize(self, loss):
        if Global.CFG.REGULARIZATION.MODE == "L1":
            l1_reg = sum([param.abs().sum() for name, param in self.model.named_parameters() if "bias" not in name])
            loss += Global.CFG.REGULARIZATION.STRENGTH * l1_reg
        elif Global.CFG.REGULARIZATION.MODE == "L2":
            l2_reg = sum([(param**2).sum() for name, param in self.model.named_parameters() if "bias" not in name])
            loss += Global.CFG.REGULARIZATION.STRENGTH * l2_reg

        return loss

    @staticmethod
    def get_lr_scheduler(scheduler_name, optimizer, scheduler_params=None):
        if scheduler_name == "LambdaLR":
            lr_lambda = lambda epoch: 0.96 * epoch
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, scheduler_name
                )(optimizer, lr_lambda=lr_lambda, **scheduler_params)
        elif scheduler_name == "MultiplicativeLR":
            Global.addRuntimeParam("factor", scheduler_params["factor"])
            try:
                del scheduler_params["factor"]
            except:
                pass
            lr_lambda = lambda epoch: Global.getRuntimeParam("factor")
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, scheduler_name
            )(optimizer, lr_lambda=lr_lambda, **scheduler_params)
        elif scheduler_name == "SequentialLR":
            schedulers = scheduler_params["schedulers"]
            sequential_schedulers = []
            for schdlr in schedulers:
                schdlr_params = schdlr["params"]
                lr_schdlr = Train.get_lr_scheduler(
                    schdlr["name"],
                    optimizer=optimizer,
                    scheduler_params=schdlr_params
                )
                sequential_schedulers.append(lr_schdlr)
            
            del scheduler_params["schedulers"]

            lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer=optimizer, schedulers=sequential_schedulers, **scheduler_params
            )
        else: 
            try:
                lr_scheduler = getattr(
                    torch.optim.lr_scheduler, scheduler_name
                    )(optimizer, **scheduler_params)
                return lr_scheduler
            except:
                if scheduler_name == "WarmUpCosineLRScheduler":
                    after_scheduler = scheduler_params["after_scheduler"]
                    after_scheduler_name = after_scheduler.NAME
                    after_scheduler_params = after_scheduler.PARAMS
                    after_lr_scheduler = Train.get_lr_scheduler(
                        scheduler_name=after_scheduler_name,
                        optimizer=optimizer,
                        scheduler_params=after_scheduler_params
                    )
                    del scheduler_params["after_scheduler"]
                    lr_scheduler = getattr(custom_lrs, scheduler_name)(
                        optimizer=optimizer, after_scheduler=after_lr_scheduler, **scheduler_params
                    )
                    return lr_scheduler

                if scheduler_name == "WarmUpLinearLRScheduler":
                    after_scheduler = scheduler_params["after_scheduler"]
                    after_scheduler_name = after_scheduler.NAME
                    after_scheduler_params = after_scheduler.PARAMS
                    after_lr_scheduler = Train.get_lr_scheduler(
                        scheduler_name=after_scheduler_name,
                        optimizer=optimizer,
                        scheduler_params=after_scheduler_params
                    )
                    del scheduler_params["after_scheduler"]
                    lr_scheduler = getattr(custom_lrs, scheduler_name)(
                        optimizer=optimizer, after_scheduler=after_lr_scheduler, **scheduler_params
                    )
                    return lr_scheduler

        return lr_scheduler

    def _setup_profiler(self):
        self.profiling_trace_path = os.path.join(Global.CFG.PROFILER.PATH, Global.CFG.PROFILER.BASENAME)
        check_dir(
            path=self.profiling_trace_path,
            create=True,
            forcedCreate=True,
            tree=True
        )

        self.profiler = torch.profiler.profile(
                activities=[Train._ACTIVITY_MAPS[i] for i in ["CPU", "CUDA"]],
                schedule=torch.profiler.schedule(**Global.CFG.PROFILER.STEPS),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiling_trace_path)
            )

    def _run_profiling(self, epoch):
        with self.profiler as prof:
            epoch_loss = self.train_for_one_epoch(epoch=epoch, profiler=prof)

        return epoch_loss

    def log_train_setting(self):

        if Global.CFG.REPEAT_AUGMENTATIONS:
            Global.LOGGER.info(f"Repeated Augmentation is enable with repetitions of {Global.CFG.REPEAT_AUGMENTATIONS_NUM_REPEATS}")
        if Global.CFG.DATA_MIXING.enabled:
            Global.LOGGER.info(f"Advanced data augmentations of cutmix and mixup is being used")
        if self.gradient_clipping is not None:
            Global.LOGGER.info(f"Gradient clipping is enabled with type: {self.gradient_clipping[0]} and threshold: {self.gradient_clipping[1]}")
        if self.gradient_accumulation:
            Global.LOGGER.info(f"Gradient accumulation is enabled to approximate batch size of: {self.gradient_accumulation_batch_size}")
        if self.exponential_moving_average is not None:
            message = ", ".join(f"{key}: {value}" for key, value in self.exponential_moving_average.items())
            Global.LOGGER.info(f"Exponential moving average is enabled while inferencing with: {message}")

    @staticmethod
    def async_train(*args, **kwargs):

        try:
            rank = args[0]
        except IndexError:
            rank = int(os.environ["LOCAL_RANK"])
        finally:
            try:
                world_size = kwargs.pop("world_size")
            except:
                world_size = args[2]
            parsed_args = kwargs.pop("args", None)
            backbone_name = parsed_args.model_name if parsed_args is not None else args[1]
        
        async_parallel_setup(rank=rank, world_size=world_size, torchrun=kwargs.pop("torchrun", False))
        cfg = setup_config(args=parsed_args)
        Global.setConfiguration(cfg)
        logger = start_logger(rank=rank, async_parallel=True, return_=True)
        Global.LOGGER = logger
        
        if Global.CFG.DEBUG is not None:
            Global.LOGGER.info(f"Running in debug mode")

        if Global.CFG.RESUME_TRAINING:
            Global.LOGGER.info(f"Resuming training process from epoch checkpoint")
            checkpoint = Checkpoint.load(model=None, name=backbone_name, checkpoint_name=None, return_checkpoint=True)

        Global.LOGGER.info(f"Instantiating {cfg.LOGGING.NAME} Architecture for classification on 1000 classes")
        model = getattr(contrastive, backbone_name)(**cfg[backbone_name].PARAMS)

        if Global.CFG.RESUME_TRAINING:
            Global.LOGGER.info(f"Loading state of model from checkpoint")
            model.load_state_dict(
                state_dict=checkpoint["model_state_dict"],
                strict=True
            )

        if cfg.USE_SYNC_BN:
            Global.LOGGER.info(f"Converting BatchNorm layers to SyncBatchNorm")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model.to(device=rank)

        model = DDP(model, device_ids=[rank])
        Global.LOGGER.info(f"{cfg.LOGGING.NAME} Architecture instantiated")

        Global.LOGGER.info(f"Instantiating Optimizer: {cfg[backbone_name].OPTIMIZER.NAME}")
        try:
            optimizer = getattr(torch.optim, cfg[backbone_name].OPTIMIZER.NAME)(model.parameters(), **cfg[backbone_name].OPTIMIZER.PARAMS)
        except:
            optimizer = getattr(optimizers, cfg[backbone_name].OPTIMIZER.NAME)(model.parameters(), **cfg[backbone_name].OPTIMIZER.PARAMS)
        Global.LOGGER.info(f"Optimizer instantiated")

        if Global.CFG.RESUME_TRAINING:
            Global.LOGGER.info(f"Loading state of optimizer from checkpoint")
            optimizer.load_state_dict(
                state_dict=checkpoint["optimizer_state_dict"]
            )

        train_dataset = ContrastiveDataset("train", transforms=cfg[backbone_name].TRANSFORMS.TRAIN, same_transforms=cfg.SAME_CONTRASTIVE_TRANSFORMS, ddp=True, debug=cfg.DEBUG)
        val_dataset = ContrastiveDataset("val", transforms=cfg[backbone_name].TRANSFORMS.VAL, same_transforms=cfg.SAME_CONTRASTIVE_TRANSFORMS, ddp=True, debug=cfg.DEBUG)

        Global.LOGGER.info(f"Intantiating data loaders for training and validation")
        data_loaders = {}
        cfg[backbone_name].DATALOADER_TRAIN_PARAMS.pop("shuffle")
        cfg[backbone_name].DATALOADER_VAL_PARAMS.pop("shuffle")

        if cfg.REPEAT_AUGMENTATIONS:
            train_sampler = RepeatAugSampler(
                dataset=train_dataset, num_replicas=world_size, rank=rank, num_repeats=cfg.REPEAT_AUGMENTATIONS_NUM_REPEATS
            )
        else:
            train_sampler = DistributedSampler(
                dataset=train_dataset, num_replicas=world_size, rank=rank
            )
        data_loaders["train"] = DataLoader(
            dataset=train_dataset, collate_fn=train_dataset.collate_fn,
            sampler=train_sampler, **cfg[backbone_name].DATALOADER_TRAIN_PARAMS
        )
        data_loaders["val"] = DataLoader(
            dataset=val_dataset, collate_fn=val_dataset.collate_fn,
            **cfg[backbone_name].DATALOADER_VAL_PARAMS
        )
        Global.LOGGER.info(f"Data loaders instantiated")

        try:
            Global.LOGGER.info(f"Instantiating Learning Rate Scheduler: {cfg[backbone_name].LR_SCHEDULER.NAME}")
            lr_scheduler = Train.get_lr_scheduler(
                scheduler_name=cfg[backbone_name].LR_SCHEDULER.NAME,
                optimizer=optimizer,
                scheduler_params=cfg[backbone_name].LR_SCHEDULER.PARAMS
            )
            Global.LOGGER.info(f"Learning Rate Scheduler instantiated")
        except:
            Global.LOGGER.info(f"No Learning Rate Scheduler specified")
            lr_scheduler = None

        if Global.CFG.RESUME_TRAINING:
            Global.LOGGER.info(f"Loading state of lr scheduler from checkpoint")
            lr_scheduler.load_state_dict(
                state_dict=checkpoint["scheduler_state_dict"],
            )

        Global.LOGGER.info(f"Instantiating Loss Function: {cfg[backbone_name].LOSS.NAME}")
        try:
            loss_function = getattr(torch.nn, cfg[backbone_name].LOSS.NAME)(**cfg[backbone_name].LOSS.PARAMS)
        except:
            loss_function = getattr(custom_loss_fns, cfg[backbone_name].LOSS.NAME)(**cfg[backbone_name].LOSS.PARAMS)
        Global.LOGGER.info(f"Loss Function instantiated")

        tb_writer = TensorboardWriter() if rank == 0 else None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Global.LOGGER.info(f"Starting training on device: {device}")
        
        train = Train(
            model=model,
            optimizer=optimizer,
            data_loaders=data_loaders,
            loss_function=loss_function,
            lr_scheduler=lr_scheduler,
            tb_writer=tb_writer,
            lr_tb_write_per_epoch=True,
            async_parallel_rank=rank,
            async_parallel=True,
            profiling=cfg.PROFILING,
            resume_epoch=0 if not Global.CFG.RESUME_TRAINING else checkpoint["epoch"] + 1,
            **cfg.TRAIN.PARAMS
        )

        train.start()