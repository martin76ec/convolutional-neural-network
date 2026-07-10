import torch.nn as nn
from torchvision import models


def get_feature_extractor(num_classes: int):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    resnet.num_classes = num_classes
    return resnet


def get_fine_tuned(num_classes: int, unfreeze_from: str = "layer4"):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in resnet.parameters():
        param.requires_grad = False

    unfreeze = False
    for name, param in resnet.named_parameters():
        if name.startswith(unfreeze_from):
            unfreeze = True
        if unfreeze:
            param.requires_grad = True

    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    resnet.num_classes = num_classes
    return resnet