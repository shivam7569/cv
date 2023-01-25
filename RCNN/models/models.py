import torch
import torch.nn as nn
from torchvision import models

from RCNN.utils.globalParams import Global

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = Global.NUM_CLASSES, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def alexnet(pretrained=True):

    if pretrained: alexnet = models.alexnet(weights='IMAGENET1K_V1')
    else: alexnet = models.alexnet(weights=None)

    num_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)

    return alexnet.to(device=Global.TORCH_DEVICE)

def svm(feature_model_path, model_name):

    model = models.alexnet(weights=None) if model_name == "alexnet" else models.vgg16(weights=None)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)

    checkpoint = torch.load(feature_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)

    return model.to(device=Global.TORCH_DEVICE)

def classifier(model_path, model_name):
    model = models.alexnet(weights=None) if model_name == "alexnet" else models.vgg16(weights=None)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model.to(device=Global.TORCH_DEVICE)

def featureModel(classifier_model_path, model_name):

    model = models.alexnet(weights=None) if model_name == "alexnet" else models.vgg16(weights=None)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)

    checkpoint = torch.load(classifier_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    feature_layers = []
    for layer in model.features.modules():
        feature_layers.append(layer)
    feature_layers = feature_layers[1:]

    avg_layers = []
    for layer in model.avgpool.modules():
        avg_layers.append(layer)

    class_layers = []
    for layer in model.classifier.modules():
        class_layers.append(layer)

    class_layers = [class_layers[1]]

    layers = feature_layers + avg_layers + [nn.Flatten()] +  class_layers
    feature_model = nn.Sequential(*layers)

    return feature_model.to(device=Global.TORCH_DEVICE)

def regressor():

    model = nn.Linear(Global.REGRESSOR_IN_FEATURES, Global.REGRESSOR_OUT_FEATURES)
    
    return model.to(device=Global.TORCH_DEVICE)


def VGG16(pretrained=False):

    if pretrained: vgg16 = models.vgg16(weights='IMAGENET1K_V1')
    else: vgg16 = models.vgg16(weights=None)

    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, Global.NUM_CLASSES)
    nn.init.kaiming_normal_(vgg16.classifier[6].weight, mode="fan_in")

    return vgg16.to(device=Global.TORCH_DEVICE)
