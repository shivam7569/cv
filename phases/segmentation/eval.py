from tqdm import tqdm
import torch
from datasets.segmentation.dataset import SegmentationDataset
from src.gpu_devices import GPU_Support
import torch.nn.functional as F
from src.metrics import SegMetrics
from utils.typing_utils import draw_confusion_matrix

class Eval:

    def __init__(
            self,
            model,
            data_loader,
            loss_function,
            tb_writer,
            async_parallel,
            async_parallel_rank,
            num_classes=81
    ):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.tb_writer = tb_writer

        self.async_parallel = async_parallel
        self.async_parallel_rank = async_parallel_rank

        self.metrics = SegMetrics(num_classes=num_classes)

    def start(self, epoch):
        self.model.eval()
        if self.tb_writer is not None: self.tb_writer.setWriter("val")

        test_image = False

        data_iterator = tqdm(self.data_loader, desc=f"Evaluating", unit="batch")
        loss = 0
        for batch in data_iterator:
            img_batch, mask_batch = batch

            if not self.async_parallel:
                if GPU_Support.support_gpu:
                    last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"

                    img_batch = img_batch.to(last_gpu_id)
                    mask_batch = mask_batch.to(last_gpu_id)
            else:
                img_batch = img_batch.to(self.async_parallel_rank, non_blocking=True)
                mask_batch = mask_batch.to(self.async_parallel_rank, non_blocking=True)

            with torch.no_grad():
                output = self.model(img_batch)
                batch_loss = self.loss_function(output, mask_batch.squeeze(1)).item()
                loss += batch_loss

                data_iterator.set_postfix(loss=batch_loss, refresh=True)
                predicted_masks = F.log_softmax(output, dim=1).exp().argmax(dim=1)

                self.metrics.update(mask_batch.squeeze(1), output)

                if not test_image and epoch % 2 == 0:
                    self.tb_writer.write("image")(image=SegmentationDataset._vizualizeBatch(batch=(img_batch, predicted_masks)), epoch=epoch+1, tag=f"Inference")
                    test_image = True

        loss /= len(self.data_loader)
        self.metrics.aggregate_metrics()
        self.metrics.log_metrics(epoch=epoch, loss=round(loss, 3))
        self.epoch_metrics = self.metrics.metrics_aggregated
        self.epoch_metrics["eval_loss"] = loss

        SegMetrics.writeMetricsToCSV()
        self.metrics.normalize_cm()
        self.tb_writer.write("figure")(figure=draw_confusion_matrix(self.metrics.normalized_confusion_matrix.cpu().numpy()), epoch=epoch+1, tag="Confusion Matrix")

        if self.tb_writer is not None:
            self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=loss, step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Accuracy", scalar_value=self.epoch_metrics["accuracy"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Precision", scalar_value=self.epoch_metrics["precision"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Recall", scalar_value=self.epoch_metrics["recall"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval F1 Score", scalar_value=self.epoch_metrics["f1_score"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval IoU Score", scalar_value=self.epoch_metrics["iou"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Dice Score", scalar_value=self.epoch_metrics["dice"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Kappa Score", scalar_value=self.epoch_metrics["kappa"], step=epoch)

        self.metrics.reset()
        if self.tb_writer is not None: self.tb_writer.setWriter("train")
