import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models as pyModels

from backbones import AlexNet
from configs.config import setup_config
from utils.global_params import Global

from datasets.classification.dataset import ClassificationDataset

if __name__ == "__main__":
    
    cfg = setup_config()
    Global.setConfiguration(cfg)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model_checkpoint = "checkpoints/AlexNet/AlexNet_epoch_45_f1_0.526.pth"
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model = AlexNet(num_classes=1000)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    model = pyModels.alexnet(weights=pyModels.AlexNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    val_dataset = ClassificationDataset("val", transforms=cfg.AlexNet.TRANSFORMS.VAL, debug=None, log=False)

    data_loader = DataLoader(
        dataset=val_dataset,
        **cfg.AlexNet.DATALOADER_VAL_PARAMS
    )

    GTS = []
    PREDS = []

    data_iterator = tqdm(data_loader, desc=f"Evaluating", unit="batch")
    for batch in data_iterator:
        img_batch, lbl_batch = batch

        img_batch = img_batch.to(device)
        lbl_batch = lbl_batch.to(device)

        output = model(img_batch)

        predicted_classes = torch.argmax(output, dim=1)

        GTS.extend(lbl_batch.tolist())
        PREDS.extend(predicted_classes.tolist())

    misclassifications = sum(1 for true, pred in zip(GTS, PREDS) if true != pred)
    total_examples = len(GTS)
    error_rate = (misclassifications / total_examples) * 100

    print(f"Error Rate: {error_rate:.2f}%")