from tqdm import tqdm
import torch
from src.gpu_devices import GPU_Support

from src.metrics import ClassificationMetrics

class Eval:

    def __init__(
            self,
            model,
            data_loader,
            loss_function,
            tb_writer,
            async_parallel,
            async_parallel_rank,
            num_classes=1000
    ):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.tb_writer = tb_writer

        self.async_parallel = async_parallel
        self.async_parallel_rank = async_parallel_rank

        self.metrics = ClassificationMetrics(num_classes=num_classes)

    def start(self, epoch):
        self.model.eval()
        if self.tb_writer is not None: self.tb_writer.setWriter("val")

        data_iterator = tqdm(self.data_loader, desc=f"Evaluating", unit="batch")
        loss = 0
        for batch in data_iterator:
            img_batch, lbl_batch = batch

            if not self.async_parallel:
                if GPU_Support.support_gpu:
                    last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"

                    img_batch = img_batch.to(last_gpu_id)
                    lbl_batch = lbl_batch.to(last_gpu_id)
            else:
                img_batch = img_batch.to(self.async_parallel_rank, non_blocking=True)
                lbl_batch = lbl_batch.to(self.async_parallel_rank, non_blocking=True)

            with torch.no_grad():
                output = self.model(img_batch)
                batch_loss = self.loss_function(output, lbl_batch).item()
                loss += batch_loss

                data_iterator.set_postfix(loss=batch_loss, refresh=True)

                predicted_classes = torch.argmax(output, dim=1)
                
                self.metrics.update(lbl_batch.cpu().numpy(), predicted_classes.cpu().numpy())

        loss /= len(self.data_loader)
        self.metrics.aggregate_metrics()
        self.metrics.log_metrics(epoch=epoch, loss=round(loss, 3))
        self.epoch_metrics = self.metrics.metrics_aggregated
        self.epoch_metrics["eval_loss"] = loss

        ClassificationMetrics.writeMetricsToCSV()

        if self.tb_writer is not None:
            self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=loss, step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Accuracy", scalar_value=self.epoch_metrics["accuracy"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Precision", scalar_value=self.epoch_metrics["precision"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Recall", scalar_value=self.epoch_metrics["recall"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval F1 Score", scalar_value=self.epoch_metrics["f1_score"], step=epoch)

        self.metrics.reset()
        if self.tb_writer is not None: self.tb_writer.setWriter("train")
