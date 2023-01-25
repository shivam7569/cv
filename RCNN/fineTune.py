from RCNN.utils.data.batch_sampler import BatchSampler
from RCNN.utils.data.finetune_dataset import FineTuneDataset
from RCNN.utils.util import check_dir
from RCNN.models.models import VGG16, AlexNet, alexnet
from RCNN.utils.globalParams import Global
import os
from time import sleep
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class FineTune:

    def __init__(self, debug=False, model_name="alexnet"):

        self.debug = debug
        self.model_name = model_name
        check_dir(Global.FINETUNE_TENSORBOARD_LOG_DIR)
        check_dir(Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/")
        self.tbWriter = SummaryWriter(Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/")
        check_dir(Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/" + "train/")
        self.trainTbWriter = SummaryWriter(
            Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/" + "train/")
        check_dir(Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/" + "val/")
        self.valTbWriter = SummaryWriter(
            Global.FINETUNE_TENSORBOARD_LOG_DIR + model_name + "/" + "val/")

    def load_data(self, transformation):
        num_workers = 33 if self.model_name == "alexnet" else 4

        data_loaders = {}
        data_sizes = {}

        for phase in ["train", "val"]:
            data_dir = os.path.join(Global.FINETUNE_DATA_DIR, phase)
            dataset = FineTuneDataset(
                data_dir, transformation[phase], phase, self.debug)

            print(
                f"\nNumber of positive and negative samples in {phase} are {dataset.get_positive_num()} and {dataset.get_negative_num()} respectively")

            data_sampler = BatchSampler(
                dataset.get_positive_num(),
                dataset.get_negative_num(),
                batch_positive=Global.FINETUNE_POSITIVE_SAMPLES,
                batch_negative=Global.FINETUNE_NEGATIVE_SAMPLES,
                shuffle=True
            )
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=Global.FINETUNE_BATCH_SIZE,
                sampler=data_sampler,
                num_workers=num_workers,
                drop_last=True
            )

            data_loaders[phase] = data_loader
            data_sizes[phase] = data_sampler.__len__()

        self.data_loaders = data_loaders
        self.data_sizes = data_sizes

    def set_device(self):
        self.device = Global.TORCH_DEVICE

    def load_model(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model = model.load_state_dict(checkpoint['state_dict'])
        optimizer = optimizer.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]

        return model, optimizer, epoch

    def resume_training(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        return model, optimizer, checkpoint["epoch"]
    
    def _get_l1_loss(self, model):
        params = torch.cat([x.view(-1) for x in model.classifier.parameters()])
        l1_regularization = 1e-4 * torch.norm(params, 1)

        return l1_regularization
    
    def train(self, model, criterion, optimizer, epochs):

        best_acc = 0.0

        model.to(self.device)

        batch_counter_train = 0
        batch_counter_val = 0
        epoch_counter = 0

        for epoch in range(epochs):

            p_mess = f"\nEpoch: {epoch+1}/{epochs}"
            print(f"{p_mess}\n{'*' * (len(p_mess) - 1)}")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                dtLoaderLen = len(self.data_loaders[phase])

                with tqdm(self.data_loaders[phase], unit="batch", total=dtLoaderLen) as tepoch:
                    for imgs, labels in tepoch:

                        tepoch.set_description(f"Epoch: {epoch+1}/{phase}")

                        imgs = imgs.to(self.device)
                        labels = labels.to(self.device).to(torch.int64)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            optimizer.zero_grad()
                            l1_loss = self._get_l1_loss(model)
                            loss += l1_loss
                            loss.backward()
                            optimizer.step()

                        step_loss = loss.item()
                        step_acc = torch.sum(
                            preds == labels.data) / imgs.shape[0]
                        running_loss += step_loss
                        running_corrects += step_acc

                        if phase == "train":

                            if batch_counter_train % 1000 == 0:

                                self.tbWriter.add_scalar(f"a_train/Step Train Loss", round(
                                    step_loss, 3), batch_counter_train)
                                self.tbWriter.add_scalar(f"a_train/Step Train Acc", round(
                                    step_acc.item(), 3), batch_counter_train)

                            batch_counter_train += 1

                        elif phase == "val":

                            if batch_counter_val % 250 == 0:

                                self.tbWriter.add_scalar(f"b_val/Step Val Loss", round(
                                    step_loss, 3), batch_counter_val)
                                self.tbWriter.add_scalar(f"b_val/Step Val Acc", round(
                                    step_acc.item(), 3), batch_counter_val)

                            batch_counter_val += 1

                        tepoch.set_postfix(loss=round(step_loss, 3), accuracy=round(step_acc.item(), 3))

                        sleep(0.1)

                epoch_loss = running_loss / dtLoaderLen
                epoch_acc = running_corrects.item() / dtLoaderLen

                print(f"{phase} loss: {round(epoch_loss, 3)}")
                print(f"{phase} acc: {round(epoch_acc, 3)}")

                if phase == "train":
                    self.trainTbWriter.add_scalar(
                        f"Epoch Loss", round(epoch_loss, 3), epoch_counter)
                    self.trainTbWriter.add_scalar(
                        f"Epoch Acc", round(epoch_acc, 3), epoch_counter)
                else:
                    self.valTbWriter.add_scalar(
                        f"Epoch Loss", round(epoch_loss, 3), epoch_counter)
                    self.valTbWriter.add_scalar(
                        f"Epoch Acc", round(epoch_acc, 3), epoch_counter)

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc

                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }

                    checkpoint_name = f"epoch_{epoch+1}_val_acc_{round(best_acc, 4)}.pt"
                    check_dir(Global.FINETUNE_CHECKPOINT_DIR)
                    check_dir(Global.FINETUNE_CHECKPOINT_DIR + self.model_name + "/")
                    torch.save(checkpoint, Global.FINETUNE_CHECKPOINT_DIR + self.model_name + "/" +
                               checkpoint_name)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        avg_grad = torch.mean(param.grad)
                        self.tbWriter.add_scalar(
                            f"c_grad_avg/{name}", avg_grad.item(), epoch_counter)
                        self.tbWriter.add_histogram(
                            f"c_grad/{name}", param.grad.cpu().numpy(), epoch_counter)
                    if param.data is not None:
                        avg_weight = torch.mean(param.data)
                        self.tbWriter.add_scalar(
                            f"d_weight_avg/{name}", avg_weight.item(), epoch_counter)
                        self.tbWriter.add_histogram(
                            f"d_weight/{name}", param.data.cpu().numpy(), epoch_counter)

            epoch_counter += 1

def performFineTuning(epochs=25, model_name="alexnet", debug=False):

    model = alexnet(pretrained=True) if model_name == "alexnet" else VGG16(pretrained=True)

    transformation_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.FancyPCA(p=0.5),
            A.GaussNoise(p=0.5),
            A.Rotate(limit=15, p=0.2),
            A.Resize(
                height=Global.IMAGE_SIZE[0], width=Global.IMAGE_SIZE[1], always_apply=True, interpolation=2, p=1),
            A.Normalize(always_apply=True, p=1),
            ToTensorV2()
        ]
    )

    transformation_val = A.Compose(
        [
            A.Resize(
                height=Global.IMAGE_SIZE[0], width=Global.IMAGE_SIZE[1], always_apply=True, p=1),
            A.Normalize(always_apply=True, p=1),
            ToTensorV2()
        ]
    )

    transformation = {}
    transformation["train"] = transformation_train
    transformation["val"] = transformation_val

    fineTune = FineTune(debug=debug, model_name=model_name)
    fineTune.load_data(transformation=transformation)
    fineTune.set_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], lr=1e-4, momentum=0.9, weight_decay=1e-4)

    fineTune.train(model, criterion, optimizer, epochs)


def loadBestFineTuneModel():
    fineTune = FineTune(debug=False)
    alexnet = AlexNet(pretrained=False)
    fineTune.load_model(alexnet, Global.BEST_FINETUNE_MODEL)

    return alexnet
