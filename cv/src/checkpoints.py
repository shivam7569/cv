import os
import torch

from cv.utils import Global
from cv.utils.os_utils import check_dir
from cv.utils import MetaWrapper

class Checkpoint(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class for checkpointing of architecures"

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_dir = os.path.join(Global.CFG.CHECKPOINT.PATH, Global.CFG.CHECKPOINT.BASENAME)

        if not Global.CFG.RESUME_TRAINING:
            check_dir(self.checkpoint_dir, create=True, forcedCreate=True, tree=Global.CFG.CHECKPOINT.TREE)

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

    @staticmethod
    def load(model, name, checkpoint_name=None, return_checkpoint=False):

        if checkpoint_name is None: checkpoint_name = name

        checkpoint_filepath = os.path.join(
            os.path.join(Global.CFG.CHECKPOINT.PATH, name),
            [i for i in os.listdir(os.path.join(Global.CFG.CHECKPOINT.PATH, name)) if checkpoint_name in i][0]
        )
        checkpoint = torch.load(checkpoint_filepath, map_location="cpu")

        if return_checkpoint: return checkpoint

        model.load_state_dict(checkpoint['model_state_dict'])

        return model
        