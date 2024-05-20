from collections import OrderedDict
import csv
import os
import numpy as np
import torch
from utils.global_params import Global
from utils.os_utils import check_file

class ClassificationMetrics:

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
        tp, fp, tn, fn = self._get_tp_fp_tn_fn()
        accuracy = (tp + tn) / (tp + fp + tn + fn + self.eps)

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

    def aggregate_metrics(self):
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

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter:
    def __init__(self, architecture, num_batches, meters, prefix="", attention_based=False):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

        self.report_path = os.path.join(
            Global.CFG.PATHS.BACKBONES, "attention" if attention_based else '' , architecture, "performance_report.txt"
        )

        if os.path.exists(self.report_path) and os.path.isfile(self.report_path):
            os.remove(self.report_path)

    def display(self, batch, write=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

        if write:
            with open(self.report_path, "a") as f:
                f.write('\t'.join(entries) + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
class SegmentationMetrics:

    instantiate_csv = False

    def __init__(self, num_classes, ignore_index=-1):
        self.num_classes = num_classes
        self.eps = 1e-6
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int64).to("cuda:0")
        self.metrics_aggregated = OrderedDict()
        self.ignore_index = ignore_index

    def update(self, lbls, pred):
        non_ignore_indices = (lbls >= 0) & (lbls < self.num_classes)
        lbls, pred = lbls[non_ignore_indices], pred[non_ignore_indices]

        indices = lbls * self.num_classes + pred
        confusion_counts = torch.bincount(indices, minlength=self.num_classes**2).reshape((self.num_classes, self.num_classes))
        
        self.confusion_matrix += confusion_counts

    def reset(self):
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int64).to("cuda:0")
        self.metrics_aggregated = OrderedDict()

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

        tp = torch.diag(self.confusion_matrix)
        fp = torch.sum(self.confusion_matrix, dim=0) - tp
        fn = torch.sum(self.confusion_matrix, dim=1) - tp
        tn = torch.sum(self.confusion_matrix) - (tp + fp + fn)
        
        return tp, fp, tn, fn
    
    def accuracy(self):
        tp, fp, tn, fn = self._get_tp_fp_tn_fn()
        accuracy = torch.clamp_min((tp + tn) / (tp + fp + tn + fn + self.eps), min=self.eps)

        return torch.round(accuracy, decimals=3)

    def precision(self):
        tp, fp, _, _ = self._get_tp_fp_tn_fn()
        precision = torch.clamp_min(tp / (tp + fp + self.eps), min=self.eps)

        return torch.round(precision, decimals=3)    

    def recall(self):
        tp, _, _, fn = self._get_tp_fp_tn_fn()
        recall = torch.clamp_min(tp / (tp + fn + self.eps), min=self.eps)

        return torch.round(recall, decimals=3)
    
    def f1_score(self):
        f1 = torch.clamp_min((2 * self.precision() * self.recall()) / (self.precision() + self.recall() + self.eps), min=self.eps)

        return torch.round(f1, decimals=3)
    
    def jaccard_index(self):
        tp, fp, _, fn = self._get_tp_fp_tn_fn()
        iou = torch.clamp_min(tp / (tp + fp + fn + self.eps), min=self.eps)

        return torch.round(iou, decimals=3)

    def dice_score(self):
        tp, fp, _, fn = self._get_tp_fp_tn_fn()
        dice = torch.clamp_min(2*tp / (2*tp + fp + fn + self.eps), min=self.eps)

        return torch.round(dice, decimals=3)
    
    def cohen_kappa(self):
        tp, _, _, _ = self._get_tp_fp_tn_fn()
        agree = tp.sum() / self.confusion_matrix.sum()
        chanceAgree = torch.sum(self.confusion_matrix.sum(dim=0) * self.confusion_matrix.sum(dim=1)) / (self.confusion_matrix.sum() ** 2)

        kappa_score = (agree - chanceAgree) / (1 - chanceAgree)

        return round(kappa_score.item(), 3)
        
    def normalize_cm(self):
        total_gts_per_class = self.confusion_matrix.sum(dim=1)[:, None]
        total_gts_per_class[total_gts_per_class == 0] = 1

        self.normalized_confusion_matrix = self.confusion_matrix / total_gts_per_class
    
    def aggregate_metrics(self):

        self.metrics_aggregated["accuracy"] = round(self.accuracy().mean().item(), 3)
        self.metrics_aggregated["precision"] = round(self.precision().mean().item(), 3)
        self.metrics_aggregated["recall"] = round(self.recall().mean().item(), 3)
        self.metrics_aggregated["f1_score"] = round(self.f1_score().mean().item(), 3)
        self.metrics_aggregated["iou"] = round(self.jaccard_index().mean().item(), 3)
        self.metrics_aggregated["dice"] = round(self.dice_score().mean().item(), 3)
        self.metrics_aggregated["kappa"] = self.cohen_kappa()

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
            if not SegmentationMetrics.instantiate_csv:
                check_file(csv_path, remove=True)
                csv_columns = Global.CFG.METRICS.SEG_COLUMNS
                with open(csv_path, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(csv_columns)
                    SegmentationMetrics.instantiate_csv = True
        else:
            SegmentationMetrics.instantiate_csv = True
        
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
        