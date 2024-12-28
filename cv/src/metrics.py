import os
import csv
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from cv.utils import Global, MetaWrapper
from cv.utils.os_utils import check_file
from cv.utils.logging_utils import AverageMeter

class ClassificationMetrics(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Evaluation class for classification models and backbones"

    instantiate_csv = False

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.eps = 1e-6
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.longlong)
        self.metrics_aggregated = OrderedDict()

    def update(self, lbls, pred):
        batch_confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.longlong)
        for pr, gt in zip(pred, lbls):
            batch_confusion_matrix[gt, pr] += 1
        self.confusion_matrix += batch_confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.longlong)
        self.metrics_aggregated = OrderedDict()
        self.epoch_labels = []
        self.epoch_preds = []

    def _get_tp_fp_tn_fn(self):
        
        # Don't be overwhelmed, this is logical to derive
        """
            tp[class_id] = confusion_matrix[class_id, class_id]
            fp[class_id] = sum(confusion_matrix[:, class_id]) - confusion_matrix[class_id, class_id] = sum(confusion_matrix[:, class_id]) - tp[class_id]
            fn[class_id] = sum(confusion_matrix[class_id, :]) - confusion_matrix[class_id, class_id] = sum(confusion_matrix[class_id, :]) - tp[class_id]
            tn[class_id] = sum(confusion_matrix) - sum(confusion_matrix[:, class_id]) - sum(confusion_matrix[class_id, :]) + confusion_matrix[class_id, class_id]
            tn[class_id] = sum(confusion_matrix) - [sum(confusion_matrix[:, class_id]) - confusion_matrix[class_id, class_id]] - [sum(confusion_matrix[class_id, :]) - confusion_matrix[class_id, class_id]] - confusion_matrix[class_id, class_id]
            tn[class_id] = sum(confusion_matrix) - (fp[class_id] + fn[class_id] + tp[class_id])
        """

        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        tn = np.sum(self.confusion_matrix) - (tp + fp + fn)
        
        return tp, fp, tn, fn
    
    def normalize_cm(self):
        total_gts_per_class = self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        total_gts_per_class[total_gts_per_class == 0] = 1

        self.normalized_confusion_matrix = self.confusion_matrix / total_gts_per_class
    
    def accuracy(self):
        tp, _, _, _ = self._get_tp_fp_tn_fn()
        accuracy = tp.sum() / self.confusion_matrix.sum()

        return np.round(accuracy, decimals=3)
    
    def precision(self):
        tp, fp, _, _ = self._get_tp_fp_tn_fn()
        precision = tp / (tp + fp + self.eps)

        return np.round(precision, decimals=3)    

    def recall(self):
        tp, _, _, fn = self._get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)

        return np.round(recall, decimals=3)
    
    def f1_score(self):
        f1 = (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + self.eps)

        return np.round(f1, decimals=3)

    def aggregate_metrics(self, **kwargs):
        
        if len(kwargs) > 0:
            class_accuracies = kwargs["accuracy"]
            class_precisions = kwargs["precision"]
            class_recalls = kwargs["recall"]
            class_f1_scores = kwargs["f1_score"]
        else:
            class_accuracies = self.accuracy()
            class_precisions = self.precision()
            class_recalls = self.recall()
            class_f1_scores = self.f1_score()

        self.metrics_aggregated["accuracy"] = np.round(np.mean(class_accuracies), decimals=3)
        self.metrics_aggregated["precision"] = np.round(np.mean(class_precisions), decimals=3)
        self.metrics_aggregated["recall"] = np.round(np.mean(class_recalls), decimals=3)
        self.metrics_aggregated["f1_score"] = np.round(np.mean(class_f1_scores), decimals=3)

    @classmethod
    def topKaccuracy(cls, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def log_metrics(self, epoch, loss=None):
        if loss is not None:
            Global.LOGGER.info(f"Evaluation loss for epoch {epoch+1}: {loss}")
        Global.LOGGER.info(f"Evaluation accuracy for epoch {epoch+1}: {self.metrics_aggregated['accuracy']}")
        Global.LOGGER.info(f"Evaluation precision for epoch {epoch+1}: {self.metrics_aggregated['precision']}")
        Global.LOGGER.info(f"Evaluation recall for epoch {epoch+1}: {self.metrics_aggregated['recall']}")
        Global.LOGGER.info(f"Evaluation f1_score for epoch {epoch+1}: {self.metrics_aggregated['f1_score']}")

        Global.METRICS["eval_loss"] = loss
        Global.METRICS["eval_accuracy"] = self.metrics_aggregated["accuracy"]
        Global.METRICS["eval_precision"] = self.metrics_aggregated["precision"]
        Global.METRICS["eval_recall"] = self.metrics_aggregated["recall"]
        Global.METRICS["eval_f1_score"] = self.metrics_aggregated["f1_score"]

    @classmethod
    def writeMetricsToCSV(cls):
        csv_path = os.path.join(Global.CFG.METRICS.PATH, Global.CFG.METRICS.NAME + ".csv")

        if not Global.CFG.RESUME_TRAINING:
            if not ClassificationMetrics.instantiate_csv:
                check_file(csv_path, remove=True)
                csv_columns = Global.CFG.METRICS.COLUMNS
                with open(csv_path, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(csv_columns)
                    ClassificationMetrics.instantiate_csv = True
        else:
            ClassificationMetrics.instantiate_csv = True
        
        with open(csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_row = [
                Global.METRICS["epoch"],
                Global.METRICS["train_loss"],
                Global.METRICS["eval_loss"],
                Global.METRICS["eval_accuracy"],
                Global.METRICS["eval_precision"],
                Global.METRICS["eval_recall"],
                Global.METRICS["eval_f1_score"]
            ]
            csv_writer.writerow(csv_row)

        Global.LOGGER.info(f"Metrics written to: {csv_path}")

class SegMetrics(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Evaluation class for segmentation models"

    instantiate_csv = False

    def __init__(self, num_classes, ignore_index=-1):

        self.num_classes = num_classes
        self.eps = 1e-7
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int64).to("cuda:0")
        self.metrics_aggregated = OrderedDict()
        self.ignore_index = ignore_index

        self.tp_meter = AverageMeter("tp")
        self.fp_meter = AverageMeter("fp")
        self.fn_meter = AverageMeter("fn")
        self.tn_meter = AverageMeter("tn")

    def update_cm(self, lbl, pred):
        non_ignore_index_mask = lbl != self.ignore_index
        lbl, pred = lbl[non_ignore_index_mask], pred[non_ignore_index_mask]

        valid_val_indices = (lbl >= 0) & (lbl < self.num_classes)
        lbl, pred = lbl[valid_val_indices], pred[valid_val_indices]

        indices = lbl * self.num_classes + pred
        confusion_counts = torch.bincount(indices, minlength=self.num_classes**2).reshape((self.num_classes, self.num_classes))
        
        self.confusion_matrix += confusion_counts

    def update(self, lbl: torch.Tensor, pred: torch.Tensor):
        """
        lbl: shape == (1, H, W)
        pred: shape == (1, C, H, W)
        """

        ignore_index_mask = lbl == self.ignore_index
        pred = F.log_softmax(pred, dim=1).exp().argmax(dim=1)

        self.update_cm(lbl.flatten(), pred.flatten())

        if ignore_index_mask.sum():
            lbl[ignore_index_mask] = self.num_classes
            pred[ignore_index_mask] = self.num_classes
            lbl = F.one_hot(lbl, self.num_classes + 1)
            pred = F.one_hot(pred, self.num_classes + 1)
            lbl = lbl[...,:-1]
            pred = pred[...,:-1]
        else:
            lbl = F.one_hot(lbl, self.num_classes)
            pred = F.one_hot(pred, self.num_classes)

        lbl = lbl.permute(0, 3, 1, 2)
        pred = pred.permute(0, 3, 1, 2)

        true_negative_mask = torch.add(lbl, pred)
        true_negative_mask = true_negative_mask == 0
        tn = torch.sum(true_negative_mask, dim=(0, 2, 3))
        tp = torch.sum(lbl & pred, dim=(0, 2, 3))
        fp = torch.clamp_min(pred - lbl, min=0).sum(dim=(0, 2, 3))
        fn = torch.clamp_min(lbl - pred, min=0).sum(dim=(0, 2, 3))

        self.tp_meter.update(tp)
        self.fp_meter.update(fp)
        self.tn_meter.update(tn)
        self.fn_meter.update(fn)

    def reset(self):
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int64).to("cuda:0")
        self.metrics_aggregated = OrderedDict()

    def normalize_cm(self):
        total_gts_per_class = self.confusion_matrix.sum(dim=1)[:, None]
        total_gts_per_class[total_gts_per_class == 0] = 1

        self.normalized_confusion_matrix = self.confusion_matrix / total_gts_per_class

    def aggregate_metrics(self):

        dice = 2*self.tp_meter.sum / (2*self.tp_meter.sum + self.fp_meter.sum + self.fn_meter.sum + self.eps)
        iou = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + self.fn_meter.sum + self.eps)
        accuracy = self.tp_meter.sum.sum()  / (self.tp_meter.sum + self.fp_meter.sum + self.tn_meter.sum + self.fn_meter.sum + self.eps)
        precision = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + self.eps)
        recall = self.tp_meter.sum / (self.tp_meter.sum + self.fn_meter.sum + self.eps)
        f1_score = 2*self.tp_meter.sum / (2*self.tp_meter.sum + self.fp_meter.sum + self.fn_meter.sum + self.eps)

        N = self.tp_meter.sum + self.fp_meter.sum + self.tn_meter.sum + self.fn_meter.sum
        p_o = (self.tp_meter.sum + self.tn_meter.sum) / N
        p_e = (((self.tp_meter.sum + self.fn_meter.sum) * (self.tp_meter.sum + self.fp_meter.sum)) / (N ** 2)) + (((self.tn_meter.sum + self.fn_meter.sum) * (self.tn_meter.sum + self.fp_meter.sum)) / (N ** 2)) 
        cohen_kappa = (p_o - p_e) / (1 - p_e)
    
        self.metrics_aggregated["accuracy"] = round(accuracy.mean().item(), 3)
        self.metrics_aggregated["precision"] = round(precision.mean().item(), 3)
        self.metrics_aggregated["recall"] = round(recall.mean().item(), 3)
        self.metrics_aggregated["f1_score"] = round(f1_score.mean().item(), 3)
        self.metrics_aggregated["iou"] = round(iou.mean().item(), 3)
        self.metrics_aggregated["dice"] = round(dice.mean().item(), 3)
        self.metrics_aggregated["kappa"] = round(cohen_kappa.mean().item(), 3)

    def log_metrics(self, epoch, loss=None):
        if loss is not None:
            Global.LOGGER.info(f"Evaluation loss for epoch {epoch+1}: {loss}")
        Global.LOGGER.info(f"Evaluation accuracy for epoch {epoch+1}: {self.metrics_aggregated['accuracy']}")
        Global.LOGGER.info(f"Evaluation precision for epoch {epoch+1}: {self.metrics_aggregated['precision']}")
        Global.LOGGER.info(f"Evaluation recall for epoch {epoch+1}: {self.metrics_aggregated['recall']}")
        Global.LOGGER.info(f"Evaluation f1_score for epoch {epoch+1}: {self.metrics_aggregated['f1_score']}")
        Global.LOGGER.info(f"Evaluation iou_score for epoch {epoch+1}: {self.metrics_aggregated['iou']}")
        Global.LOGGER.info(f"Evaluation dice_score for epoch {epoch+1}: {self.metrics_aggregated['dice']}")
        Global.LOGGER.info(f"Evaluation kappa_score for epoch {epoch+1}: {self.metrics_aggregated['kappa']}")

        Global.METRICS["eval_loss"] = loss
        Global.METRICS["eval_accuracy"] = self.metrics_aggregated["accuracy"]
        Global.METRICS["eval_precision"] = self.metrics_aggregated["precision"]
        Global.METRICS["eval_recall"] = self.metrics_aggregated["recall"]
        Global.METRICS["eval_f1_score"] = self.metrics_aggregated["f1_score"]
        Global.METRICS["eval_iou_score"] = self.metrics_aggregated["iou"]
        Global.METRICS["eval_dice_score"] = self.metrics_aggregated["dice"]
        Global.METRICS["eval_kappa_score"] = self.metrics_aggregated["kappa"]

    @classmethod
    def writeMetricsToCSV(cls):
        csv_path = os.path.join(Global.CFG.METRICS.PATH, Global.CFG.METRICS.NAME + ".csv")

        if not Global.CFG.RESUME_TRAINING:
            if not SegMetrics.instantiate_csv:
                check_file(csv_path, remove=True)
                csv_columns = Global.CFG.METRICS.SEG_COLUMNS
                with open(csv_path, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(csv_columns)
                    SegMetrics.instantiate_csv = True
        else:
            SegMetrics.instantiate_csv = True
        
        with open(csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_row = [
                Global.METRICS["epoch"],
                Global.METRICS["train_loss"],
                Global.METRICS["eval_loss"],
                Global.METRICS["eval_accuracy"],
                Global.METRICS["eval_precision"],
                Global.METRICS["eval_recall"],
                Global.METRICS["eval_f1_score"],
                Global.METRICS["eval_iou_score"],
                Global.METRICS["eval_dice_score"],
                Global.METRICS["eval_kappa_score"]
            ]
            csv_writer.writerow(csv_row)

        Global.LOGGER.info(f"Metrics written to: {csv_path}")
