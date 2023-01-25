import os
import random
from time import sleep
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from RCNN.models.models import svm
from RCNN.utils.data.batch_sampler import BatchSampler
from RCNN.utils.data.classifier_dataset import ClassifierDataset
from RCNN.utils.data.hard_negative_mining import HardNegativeMiningDataset
from RCNN.utils.globalParams import Global
from RCNN.utils.util import check_dir

from tensorboardX import SummaryWriter

import warnings
warnings.simplefilter("ignore", UserWarning)

class SVM:

    def __init__(self, debug=False):
        self.debug = debug
        check_dir(Global.SVM_TENSORBOARD_LOG_DIR)
        self.tbWriter = SummaryWriter(Global.SVM_TENSORBOARD_LOG_DIR)
        check_dir(Global.SVM_TENSORBOARD_LOG_DIR + "/train/")
        self.trainTbWriter = SummaryWriter(
            Global.SVM_TENSORBOARD_LOG_DIR + "/train/")
        check_dir(Global.SVM_TENSORBOARD_LOG_DIR + "/val/")
        self.valTbWriter = SummaryWriter(
            Global.SVM_TENSORBOARD_LOG_DIR + "/val/")
        
    def load_data(self, transformation):

        data_loaders = {}
        data_sizes = {}
        remain_negative_list = []

        for phase in ["train", "val"]:
            data_dir = os.path.join(Global.CLASSIFIER_DATA_DIR, phase)
            dataset = ClassifierDataset(data_dir, transformation[phase], phase, self.debug)

            if phase == "train":

                positive_list = dataset.get_positives()
                negative_list = dataset.get_negatives()

                init_negative_idxs = set(random.sample(range(len(negative_list)), len(positive_list)))
                remain_negative_idx = set(range(len(negative_list))).difference(init_negative_idxs)

                init_negative_list = [
                        negative_list[idx] 
                        for idx in init_negative_idxs
                    ]
                remain_negative_list = [
                        negative_list[idx] 
                        for idx in remain_negative_idx
                    ]
                
                dataset.set_negative_list(init_negative_list)
                data_loaders["remain"] = remain_negative_list

            data_sampler = BatchSampler(
                dataset.get_positive_num(),
                dataset.get_negative_num(),
                batch_positive=Global.SVM_POSITIVE_SAMPLES,
                batch_negative=Global.SVM_NEGATIVE_SAMPLES,
                shuffle=True
            )
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=Global.SVM_BATCH_SIZE,
                sampler=data_sampler,
                num_workers=8,
                drop_last=True
            )

            data_loaders[phase] = data_loader
            data_sizes[phase] = len(data_sampler)

        self.data_loaders = data_loaders
        self.data_sizes = data_sizes
    
    def _hinge_loss(self, outputs, labels):

        batch_size = len(labels)
        corrects = outputs[range(batch_size), labels].unsqueeze(0).T

        margin = 1.0
        margins = outputs - corrects + margin

        loss = torch.sum(torch.max(margins, 1)[0]) / batch_size

        return loss
    
    def set_device(self):
        self.device = Global.TORCH_DEVICE

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
                    for imgs, labels, _ in tepoch:

                        tepoch.set_description(f"Epoch: {epoch+1}/{phase}")

                        imgs = imgs.to(self.device)
                        labels = labels.to(self.device).to(torch.int64)

                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        step_loss = loss.item()
                        step_acc = torch.sum(
                            preds == labels.data) / imgs.shape[0]
                        running_loss += step_loss
                        running_corrects += step_acc

                        if phase == "train":

                            if batch_counter_train % 100 == 0:

                                self.tbWriter.add_scalar(f"a_train/Step Train Loss", round(
                                    step_loss, 3), batch_counter_train)
                                self.tbWriter.add_scalar(f"a_train/Step Train Acc", round(
                                    step_acc.item(), 3), batch_counter_train)

                            batch_counter_train += 1

                        elif phase == "val":

                            if batch_counter_val % 100 == 0:

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
                    check_dir(Global.CLASSIFIER_CHECKPOINT_DIR)
                    torch.save(checkpoint, Global.CLASSIFIER_CHECKPOINT_DIR +
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

            train_dataset = self.data_loaders["train"].dataset
            remain_negative_list = self.data_loaders["remain"]
            jpeg_images = train_dataset.get_images()
            transform = train_dataset.get_transform()

            with torch.set_grad_enabled(mode=False):

                print("\nHard Negative Mining")

                remain_dataset = HardNegativeMiningDataset(remain_negative_list, jpeg_images, transform, "train")
                remain_data_loader = DataLoader(
                    remain_dataset,
                    batch_size=Global.SVM_BATCH_SIZE,
                    num_workers=8,
                    drop_last=True
                )

                negative_list = train_dataset.get_negatives()
                add_negative_list = self.data_loaders.get("add_negative", [])

                running_corrs = 0

                dtLoaderLen = len(remain_data_loader)
                with tqdm(remain_data_loader, unit="batch", total=dtLoaderLen) as tepoch:
                    for imgs, labels, cached_dicts in tepoch:

                        tepoch.set_description(f"mining")

                        imgs = imgs.to(self.device)
                        labels = labels.to(self.device).to(torch.int64)

                        optimizer.zero_grad()
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)

                        running_corrs += torch.sum(preds == labels.data)

                        hard_negative_list, _ = remain_dataset.get_hard_negatives(preds.cpu().numpy(), cached_dicts)
                        remain_dataset.add_hard_negatives(hard_negative_list, negative_list, add_negative_list)

                    remain_acc = running_corrs.double() / len(remain_negative_list)
                    print('Remain Negative Size: {}, acc: {:.3f}'.format(len(remain_negative_list), remain_acc))

                    train_dataset.set_negative_list(negative_list)

                    new_sampler = BatchSampler(
                        train_dataset.get_positive_num(),
                        train_dataset.get_negative_num(),
                        Global.SVM_POSITIVE_SAMPLES, Global.SVM_NEGATIVE_SAMPLES,
                        shuffle=True
                    )

                    self.data_loaders["train"] = DataLoader(
                        train_dataset, batch_size=Global.SVM_BATCH_SIZE,
                        sampler=new_sampler, num_workers=8, drop_last=True
                    )
                    self.data_loaders['add_negative'] = add_negative_list
                    self.data_sizes['train'] = len(new_sampler)



def trainSVM(feature_model_path, epochs=25, debug=False):

    print()
    model = svm(feature_model_path, model_name="vgg16")
    
    transformation_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
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

    classifier = SVM(debug=debug)
    classifier.load_data(transformation=transformation)
    classifier.set_device()

    criterion = classifier._hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    classifier.train(model, criterion, optimizer, epochs)
