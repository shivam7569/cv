import time
import torch
import backbones
from src.gpu_devices import GPU_Support
from src.metrics import AverageMeter, ClassificationMetrics, ProgressMeter
from src.checkpoints import Checkpoint
import configs
from torchvision import transforms as T
from torch.utils.data import DataLoader
from configs.config import setup_config
from datasets.classification.dataset import ClassificationDataset
from utils.global_params import Global

class Evaluate:

    __ATTENTION_MODELS__ = [
        "ViT", "DeiT"
    ]

    def __init__(self, architecture):
        self.architecture = architecture
        
        self._prepare()
        self._device()
        self._loadDataloader()
        self._loadModel()
        self._loadCriterion()

    def _loadCriterion(self):
        self.criterion = getattr(
            torch.nn, self.cfg[self.architecture].LOSS.NAME
        )(**self.cfg[self.architecture].LOSS.PARAMS)

    def _loadModel(self):

        try:
            model_params = self.cfg[self.architecture].PARAMS
        except:
            model_params = {
                "num_classes": 1000,
                "in_channels": 3
            }

        model = getattr(backbones, self.architecture)(**model_params)
        model = Checkpoint.load(model, self.architecture)
        model.eval()
        model.to(self.device)

        self.model = model

    def _device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _prepare(self):
        cfg = setup_config(default=True)
        GPU_Support.set_gpu_devices("0", log=False)

        getattr(configs, ''.join([self.architecture, "Config"]))(cfg)

        self.cfg = cfg
        Global.setConfiguration(cfg)

    def _loadDataloader(self):
        dataset = ClassificationDataset(
            phase="val",
            ddp=False, debug=None, log=False, standalone=True,
            transforms=T.Compose(ClassificationDataset.parseTransforms(self.cfg[self.architecture].TRANSFORMS.VAL))
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=False, collate_fn=dataset.collate_fn
        )

        self.dataloader = dataloader

    def validate(self):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            self.architecture,
            len(self.dataloader),
            [batch_time, losses, top1, top5],
            prefix='Evaluation: ',
            attention_based=True if self.architecture in Evaluate.__ATTENTION_MODELS__ else False)
        
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.dataloader):
                images = images.cuda()
                target = target.cuda()

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = ClassificationMetrics.topKaccuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:
                    progress.display(i, write=True)

            with open(progress.report_path, "a") as f:
                f.write("\n" + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5) + "\n")
            
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg


metric = Evaluate(architecture="ViT")
metric.validate()