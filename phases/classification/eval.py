from tqdm import tqdm
import torch

from src.metrics import ClassificationMetrics

class Eval:

    def __init__(
            self,
            model,
            data_loader,
            device,
            loss_function,
            tb_writer=None,
            num_classes=1000
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.loss_function = loss_function
        self.tb_writer = tb_writer

        self.metrics = ClassificationMetrics(num_classes=num_classes)

    def start(self, epoch=None):
        self.model.eval()
        if self.tb_writer is not None: self.tb_writer.setWriter("val")

        data_iterator = tqdm(self.data_loader, desc=f"Evaluating", unit="batches")
        loss = 0
        for batch in data_iterator:
            img_batch, lbl_batch = batch
            img_batch = img_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)

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

        ClassificationMetrics.writeMetricsToCSV()

        if self.tb_writer is not None:
            self.tb_writer.write("scaler")(scalar_name="Loss", scalar_value=loss, step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Accuracy", scalar_value=self.epoch_metrics["accuracy"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Precision", scalar_value=self.epoch_metrics["precision"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval Recall", scalar_value=self.epoch_metrics["recall"], step=epoch)
            self.tb_writer.write("scaler")(scalar_name="Eval F1 Score", scalar_value=self.epoch_metrics["f1_score"], step=epoch)

        self.metrics.reset()
