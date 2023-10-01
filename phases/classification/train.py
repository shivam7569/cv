from matplotlib.pyplot import sca
import numpy as np
import torch
from tqdm import tqdm
from src.checkpoints import Checkpoint
from global_params import Global

from phases.classification.eval import Eval

class Train:

    def __init__(
            self,
            model,
            optimizer,
            data_loaders,
            device,
            loss_function,
            lr_scheduler=None,
            tb_writer=None,
            epochs=100,
    ):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.epochs = epochs
        self.device = device
        self.tb_writer = tb_writer

        self.train_loader = data_loaders["train"]
        self.val_loader = data_loaders["val"]

        self.evaluation = Eval(
            model=model,
            data_loader=data_loaders["val"],
            device=device,
            loss_function=loss_function,
            tb_writer=tb_writer
        )

        self.checkpointer = Checkpoint(
            model, optimizer, lr_scheduler
        )

    def start(self):
        f1_score = np.NINF
        for epoch in range(self.epochs):

            if self.tb_writer is not None: self.tb_writer.setWriter("train")
            epoch_loss = self.train_for_one_epoch(self.train_loader, epoch)
            if self.tb_writer is not None: self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=epoch_loss, step=epoch)

            Global.METRICS["epoch"] = epoch
            Global.METRICS["train_loss"] = epoch_loss
            self.evaluation.start(epoch=epoch)

            if self.evaluation.epoch_metrics["f1_score"] > f1_score:
                f1_score = self.evaluation.epoch_metrics["f1_score"]
                checkpoint_name = Global.CFG.CHECKPOINT.BASENAME + f"_epoch_{epoch+1}_f1_{f1_score}.pth"
                Global.LOGGER.info(f"Saving checkpoint for epoch {epoch+1}")
                self.checkpointer.save(
                    epoch=epoch,
                    chkp_name=checkpoint_name
                )

            Global.resetEpochMetrics()

            if self.lr_scheduler is not None:
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(self.evaluation.epoch_metrics["eval_loss"])
                    if self.tb_writer is not None: 
                        current_lr = self.optimizer.param_groups[0]['lr']
                        self.tb_writer.write("scaler")(scalar_name="Learning Rate", scalar_value=current_lr, step=epoch)

    def train_for_one_epoch(self, train_loader, epoch):

        self.model.train()
        data_iterator = tqdm(train_loader, desc=f"Training: Epoch {epoch+1}", unit="batches")
        
        epoch_loss = 0
        for batch in data_iterator:
            img_batch, lbl_batch = batch
        
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

            if self.tb_writer is not None: self.tb_writer.write("graph")(model=self.model, input_to_model=img_batch)

            output = self.model(img_batch)
            loss = self.loss_function(output, lbl_batch)

            if Global.CFG.REGULARIZATION.MODE in ["L1", "L2"]:
                if Global.CFG.REGULARIZATION.MODE == "L1":
                    l1_reg = sum([param.abs().sum() for name, param in self.model.named_parameters() if "bias" not in name])
                    loss += Global.CFG.REGULARIZATION.STRENGTH * l1_reg
                elif Global.CFG.REGULARIZATION.MODE == "L2":
                    l2_reg = sum([(param**2).sum() for name, param in self.model.named_parameters() if "bias" not in name])
                    loss += Global.CFG.REGULARIZATION.STRENGTH * l2_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            data_iterator.set_postfix(loss=loss.item(), refresh=True)

        epoch_loss /= len(train_loader)
        Global.LOGGER.info(f"\nTraining loss for epoch {epoch+1}: {round(epoch_loss, 3)}")
            
        return epoch_loss