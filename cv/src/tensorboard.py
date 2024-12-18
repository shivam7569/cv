import os
from torch.utils.tensorboard import SummaryWriter

from cv.utils import Global
from cv.utils.os_utils import check_dir
from cv.utils import MetaWrapper

class TensorboardWriter(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Tensorboard logging class, used across the package"

    def __init__(self, subdir=''):
        summary_path = os.path.join(Global.CFG.TENSORBOARD.PATH, Global.CFG.TENSORBOARD.BASENAME, subdir)
        
        train_summary_path = os.path.join(summary_path, "train")
        val_summary_path = os.path.join(summary_path, "val")

        if not Global.CFG.RESUME_TRAINING:
            check_dir(summary_path, create=True, forcedCreate=True, tree=True)
            check_dir(train_summary_path, create=True, forcedCreate=True)
            check_dir(val_summary_path, create=True, forcedCreate=True)

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
        elif summary_type == "figure":
            return self._write_figure_summary
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

    def _write_image_summary(self, image, epoch, tag=None):
        self.writer.add_image(
            tag=f"Epoch_{epoch}" if tag is None else tag,
            img_tensor=image,
            global_step=epoch
        )

    def _write_figure_summary(self, figure, epoch, tag=None):
        self.writer.add_figure(
            tag=f"Epoch_{epoch}" if tag is None else tag,
            figure=figure,
            global_step=epoch
        )
