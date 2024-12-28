import os
import torch
import shutil
import random
import datetime
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cv.configs.config import get_cfg
from cv.utils.os_utils import check_dir
from cv.utils import MetaWrapper, Global
from cv.src.gpu_devices import GPU_Support
from cv.datasets import ClassificationDataset
from cv.src.metrics import ClassificationMetrics
from cv.src.tensorboard import TensorboardWriter


class KnnEval(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Eval Class based on KNN Classifier"
    
    def __init__(
            self,
            model: nn.Module,
            data_loader: DataLoader,
            loss_function: nn.Module,
            tb_writer: TensorboardWriter,
            async_parallel: bool,
            async_parallel_rank: int,
            num_classes: int = 1000
    ):
        
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.tb_writer = tb_writer
        self.async_parallel = async_parallel
        self.async_parallel_rank = async_parallel_rank

        self.metrics = ClassificationMetrics(num_classes=num_classes)

    def start(self, epoch):

        self.feature_dir = os.path.join(get_cfg().TEMP_DIR, "_".join(str(datetime.datetime.now()).split(" ")))
        check_dir(self.feature_dir, create=True, forcedCreate=True, tree=False)

        self.model.eval()
        if self.tb_writer is not None: self.tb_writer.setWriter("val")

        data_iterator = tqdm(self.data_loader, desc=f"Generating Features", unit="batch")
        loss = 0

        features = []
        annotations = []

        random_sample = {}
        random_index = random.randint(0, len(self.data_loader) - 1)

        for idx, batch in enumerate(data_iterator):

            img_batch, labels = batch

            if not self.async_parallel:
                if GPU_Support.support_gpu:
                    last_gpu_id = f"cuda:{GPU_Support.support_gpu - 1}"

                    img_batch = img_batch.to(last_gpu_id)
            else:
                img_batch = img_batch.to(self.async_parallel_rank, non_blocking=True)

            with torch.no_grad():
                output = self.model(img_batch)
                batch_loss = self.loss_function(output).item()
                loss += batch_loss

                data_iterator.set_postfix(loss=batch_loss, refresh=True)

                output = output[:labels.size(0), ...]
                features.append(output)
                annotations.append(labels)

            if (random_index == idx) and (len(random_sample) == 0):
                random_sample["images"] = batch[0][:labels.size(0), ...]
                random_sample["labels"] = batch[1]
                random_sample["features"] = output

        loss /= len(self.data_loader)
        features = torch.cat(features, dim=0)
        annotations = torch.cat(annotations)    

        self.save_npy(features=features, labels=annotations)
        knn = self.fit_knn()

        random_sample["predictions"] = knn.predict(random_sample["features"].cpu().detach().numpy())
        if epoch % 10 == 0:
            self.tb_writer.write("image")(image=ClassificationDataset._vizualizeBatch(batch=(random_sample["images"], random_sample["predictions"])), epoch=epoch+1, tag="Inference")

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

        shutil.rmtree(self.feature_dir)

    def save_npy(self, **kwargs):
        for k, v in kwargs.items():
            np.save(os.path.join(self.feature_dir, f"{k}.npy"), v.cpu().detach().numpy())

    def _find_best_k(self):

        Global.LOGGER.info(f"Finding best value for number of neighbours for KNN")

        X = np.load(os.path.join(self.feature_dir, "features.npy"))
        y = np.load(os.path.join(self.feature_dir, "labels.npy"))

        k_range = range(1, 11)
        k_range_iterator = tqdm(range(1, 11), desc=f"Finding best k", unit="iter")
        cross_val_f1_scores = []

        for k in k_range_iterator:

            k_range_iterator.set_postfix(k=k, refresh=True)

            knn = KNeighborsClassifier(n_neighbors=k)
            f1 = cross_val_score(knn, X, y, cv=5, scoring="f1_micro")
            cross_val_f1_scores.append(f1.mean())

        best_k = k_range[np.argmax(cross_val_f1_scores)]

        return best_k

    def fit_knn(self):

        Global.LOGGER.info(f"Fitting KNN on extracted features")

        if not hasattr(self, "knn_k"):
            self.knn_k = self._find_best_k()

        X = np.load(os.path.join(self.feature_dir, "features.npy"))
        y = np.load(os.path.join(self.feature_dir, "labels.npy"))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=self.knn_k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="micro")
        precision = precision_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")

        self.metrics.aggregate_metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

        return knn
