import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from global_params import Global
from utils.os_utils import check_dir

class TensorboardWriter:

    def __init__(self):
        summary_path = os.path.join(Global.CFG.TENSORBOARD.PATH, Global.CFG.TENSORBOARD.BASENAME)
        check_dir(summary_path, create=True)

        train_summary_path = os.path.join(summary_path, "train")
        check_dir(train_summary_path, create=True)
        val_summary_path = os.path.join(summary_path, "val")
        check_dir(val_summary_path, create=True)

        self.train_writer = SummaryWriter(
            log_dir=train_summary_path,
            flush_secs=30
        )
        self.val_writer = SummaryWriter(
            log_dir=val_summary_path,
            flush_secs=30
        )

        self.graph_written = False

    def setWriter(self, whichOne):
        self.writer = self.train_writer if whichOne == "train" else self.val_writer

    def write(self, summary_type="scaler"): # many to be implemented yet
        if summary_type == "scaler":
            return self._write_scaler_summary
        elif summary_type == "image":
            return self._write_image_summary
        elif summary_type == "histogram":
            return self._write_histogram_summary
        elif summary_type == "graph":
            return self._write_graph_summary
        elif summary_type == "audio":
            return self._write_audio_summary
        elif summary_type == "text":
            return self._write_text_summary
        else:
            raise ValueError(f"Invalid summary type: {summary_type}")
        
    def _write_scaler_summary(self, scalar_name, scalar_value, step):
        self.writer.add_scalar(
            tag=scalar_name,
            scalar_value=scalar_value,
            global_step=step
        )

    def _write_graph_summary(self, model, input_to_model):
        if not self.graph_written:
            self.writer.add_graph(model, input_to_model)
            self.graph_written = True
