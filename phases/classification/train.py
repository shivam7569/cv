import os
import numpy as np
import torch
from tqdm import tqdm
from src.gpu_devices import GPU_Support
from src.checkpoints import Checkpoint
from utils.global_params import Global
import src.losses as custom_loss_fns
from src import custom_lrs

from phases.classification.eval import Eval
from utils.os_utils import check_dir

class Train:

    _ACTIVITY_MAP = {
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
            epochs=100,
            phase_dependent=False,
            custom_loss=None,
            gradient_clipping=None,
            profiling=None
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.loss_function = loss_function
        self.epochs = epochs
        self.tb_writer = tb_writer
        self.phase_dependent = phase_dependent
        self.custom_loss = custom_loss
        self.gradient_clipping = gradient_clipping

        self.train_loader = data_loaders["train"]
        self.val_loader = data_loaders["val"]

        self.profiling = profiling
        if profiling: self._setup_profiler()

        self.evaluation = Eval(
            model=model,
            data_loader=data_loaders["val"],
            loss_function=loss_function,
            tb_writer=tb_writer
        )

        self.checkpointer = Checkpoint(
            model, optimizer, lr_scheduler
        )

        self.graph_written = False

    def start(self):
        f1_score = np.NINF
        for epoch in range(self.epochs):

            if self.tb_writer is not None: self.tb_writer.setWriter("train")

            if self.profiling:
                epoch_loss = self._run_profiling(self.train_loader, epoch)
            else:
                epoch_loss = self.train_for_one_epoch(self.train_loader, epoch)

            if self.tb_writer is not None: self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=epoch_loss, step=epoch)

            Global.METRICS["epoch"] = epoch
            Global.METRICS["train_loss"] = epoch_loss
            self.evaluation.start(epoch=epoch)

            if self.evaluation.epoch_metrics["f1_score"] > f1_score:
                f1_score = self.evaluation.epoch_metrics["f1_score"]
                self._mark_checkpoint(epoch=epoch, f1_score=f1_score, epoch_chkpt=False)
            if Global.CFG.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS:
                self._mark_checkpoint(epoch=None, f1_score=None, epoch_chkpt=True)

            Global.resetEpochMetrics()

            if self.lr_scheduler is not None:
                self._lr_scheduling(epoch_based=True, epoch=epoch)

            if self.tb_writer is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.write("scaler")(scalar_name="Learning Rate", scalar_value=current_lr, step=epoch)

    def train_for_one_epoch(self, train_loader, epoch):

        self.model.train()
        num_iterations = len(train_loader)
        data_iterator = tqdm(train_loader, desc=f"Training: Epoch {epoch+1}", unit="batch")
        
        epoch_loss = 0
        for idx, batch in enumerate(data_iterator):
            img_batch, lbl_batch = batch

            if GPU_Support.support_gpu:
                last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"
        
                img_batch = img_batch.to(last_gpu_id)
                lbl_batch = lbl_batch.to(last_gpu_id)

            if self.tb_writer is not None and not self.graph_written:
                self.tb_writer.write("graph")(model=self.model, input_to_model=img_batch)
                self.graph_written = True

            if self.phase_dependent:
                output = self.model(img_batch, phase="training")
            else:
                output = self.model(img_batch)

            if self.custom_loss:
                loss_fn = getattr(custom_loss_fns, self.custom_loss)
                loss = loss_fn(output, lbl_batch, primitive_loss_fn=self.loss_function)
            else:
                loss = self.loss_function(output, lbl_batch)

            if Global.CFG.REGULARIZATION.MODE in ["L1", "L2"]:
                loss = self._regularize(loss=loss)

            self.optimizer.zero_grad()
            loss.backward()

            self._gradient_clipping()

            self.optimizer.step()

            epoch_loss += loss.item()

            if self.lr_scheduler is not None:
                batch_iteration = idx/num_iterations if self.lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts" else idx
                self._lr_scheduling(epoch_based=False, epoch=epoch, batch_iteration=batch_iteration)

            data_iterator.set_postfix(loss=loss.item(), refresh=True)

        epoch_loss /= len(train_loader)
        Global.LOGGER.info(f"\nTraining loss for epoch {epoch+1}: {round(epoch_loss, 3)}")
            
        return epoch_loss

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
            self.checkpointer.save(None, "epoch_checkpoint.pth", overwrite=False)

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
            if self.lr_scheduler.__class__.__name__ == "WarmUpPolynomialLRScheduler":
                self.lr_scheduler.step()                
        else:
            if self.lr_scheduler.__class__.__name__ == "CyclicLR":
                self.lr_scheduler.step()
            if self.lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
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
            lr_lambda = lambda epoch: scheduler_params["factor"]
            del scheduler_params["factor"]
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
                if scheduler_name == "WarmUpPolynomialLRScheduler":
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
                activities=[Train._ACTIVITY_MAP[i] for i in ["CPU", "CUDA"]],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=2
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiling_trace_path)
            )

    def _run_profiling(self, train_loader, epoch):
        with self.profiler as prof:
            epoch_loss = self.train_for_one_epoch(train_loader, epoch)
            prof.step()

        return epoch_loss