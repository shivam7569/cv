import os
import torch

from global_params import Global
from utils.os_utils import check_dir

class Checkpoint:

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_dir = os.path.join(Global.CFG.CHECKPOINT.PATH, Global.CFG.CHECKPOINT.BASENAME)
        check_dir(self.checkpoint_dir, create=True)

    def save(self, epoch, chkp_name, overwrite=True):
        chkpt_path = os.path.join(self.checkpoint_dir, chkp_name) 
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else {},
            "epoch": epoch,
        }, chkpt_path)
        if epoch is None:
            Global.LOGGER.info(f"Checkpoint saved for epoch at: {chkpt_path}")
        else:
            Global.LOGGER.info(f"Checkpoint saved for epoch {epoch+1} at: {chkpt_path}")

        if overwrite:
            Global.LOGGER.info(f"Deleting old checkpoints")
            chkpts = [i for i in os.listdir(self.checkpoint_dir) if i != chkp_name]
            for chkpt in chkpts:
                os.remove(os.path.join(self.checkpoint_dir, chkpt))
        