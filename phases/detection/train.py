import numpy as np
import torch
from tqdm import tqdm
from src.gpu_devices import GPU_Support
from src.checkpoints import Checkpoint
from utils.global_params import Global
import src.losses as custom_lss_fns

from phases.detection.eval import Eval
from utils.pytorch_utils import numpy2tensor

class Train:

    def __init__(
            self,
            model,
            optimizer,
            data_loaders,
            loss_function,
            algorithm,
            lr_scheduler=None,
            tb_writer=None,
            phase_dependent=None,
            epochs=100,
            custom_loss=None,
            gradient_clipping=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.algorithm = algorithm
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.epochs = epochs
        self.tb_writer = tb_writer
        self.custom_loss = custom_loss
        self.phase_dependent = phase_dependent
        self.gradient_clipping = gradient_clipping

        self.train_loader = data_loaders["train"]
        self.val_loader = data_loaders["val"]

        self.evaluation = Eval(
            model=model,
            algorithm=algorithm,
            data_loader=data_loaders["val"],
            loss_function=loss_function,
            tb_writer=tb_writer
        )

        self.checkpointer = Checkpoint(
            model=model, optimizer=optimizer, scheduler=lr_scheduler
        )

        self.graph_written = False

    def start(self):
        mAP_score = np.NINF

        for epoch in range(self.epochs):
            if self.tb_writer is not None: self.tb_writer.setWriter("train")
            epoch_loss = self.train_for_one_epoch(self.train_loader, epoch)
            if self.tb_writer is not None: self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=epoch_loss, step=epoch)

            Global.METRICS["epoch"] = epoch
            Global.METRICS["train_loss"] = epoch_loss

            # self.evaluation.start(epoch=epoch)



    def train_for_one_epoch(self, train_loader, epoch):
        
        self.model.train()
        data_iterator = tqdm(train_loader, desc=f"Training: Epoch {epoch+1}", unit="batch")

        epoch_loss = 0
        for batch in data_iterator:
            model_inputs = []

            if self.algorithm == "FastRCNN":
                img_batch, rois = batch
                coordinates = rois[:, :, :4]
                target = numpy2tensor(rois[:, :, 4:])

                model_inputs.append(img_batch)
                model_inputs.append(coordinates)

                # if GPU_Support.support_gpu:
                #     last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"
                    
                #     img_batch = img_batch.to(last_gpu_id)
                #     target = target.to(last_gpu_id)
            else:
                pass # to be developed as per new algorithms gets developed

            if self.phase_dependent:
                output = self.model(*model_inputs, phase="training")
            else:
                output = self.model(*model_inputs)

            if self.custom_loss:
                loss_fn = getattr(custom_lss_fns, self.custom_loss)
                loss = loss_fn(output, target, primitive_loss_fn=self.loss_function)
            else:
                loss = self.loss_function(output, target)

            # if Global.CFG.REGULARIZATION.MODE in ["L1", "L2"]:
            #     loss = self._regularize(loss=loss)

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), clip_value=self.gradient_clipping)
            
            self.optimizer.step()

            epoch_loss += loss.item()

            if self.lr_scheduler is not None:
                self._lr_scheduling(epoch_based=False)
            
            data_iterator.set_postfix(loss=loss.item(), refresh=True)

        epoch_loss /= len(train_loader)
        Global.LOGGER.info(f"\nTraining loss for epoch {epoch+1}: {round(epoch_loss, 3)}")

        return epoch_loss

    def _mark_checkpoint(self, epoch, mAP, epoch_chkpt):
        if not epoch_chkpt:
            if '/' in Global.CFG.CHECKPOINT.BASENAME:
                    basename = Global.CFG.CHECKPOINT.BASENAME.split("/")[-1]
            else:
                basename = Global.CFG.CHECKPOINT.BASENAME
            checkpoint_name = basename + f"_epoch_{epoch+1}_mAP_{mAP}.pth"
            Global.LOGGER.info(f"Saving metric checkpoint for epoch {epoch+1}")
            self.checkpointer.save(
                epoch=epoch,
                chkp_name=checkpoint_name,
                overwrite=True
            )
        else:
            self.checkpointer.save(None, "epoch_checkpoint.pth", overwrite=False)

    def _lr_scheduling(self, epoch_based, epoch=None):
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
            
        else:
            if self.lr_scheduler.__class__.__name__ == "CyclicLR":
                self.lr_scheduler.step()

    def _regularize(self, loss):
        if Global.CFG.REGULARIZATION.MODE == "L1":
            l1_reg = sum([param.abs().sum() for name, param in self.model.parameters() if "bias" not in name])
            loss += Global.CFG.REGULARIZATION.STRENGTH * l1_reg
        elif Global.CFG.REGULARIZATION.MODE == "L2":
            l2_reg = sum([(param**2).sum() for name, param in self.model.parameters() if "bias" not in name])
            loss += Global.CFG.REGULARIZATION.STRENGTH * l2_reg
        
        return loss