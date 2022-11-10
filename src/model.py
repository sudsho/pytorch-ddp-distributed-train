"""Model factory. Currently just torchvision resnets."""
import torch.nn as nn
from torchvision import models


def build_model(arch="resnet50", num_classes=10, pretrained=False):
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
    else:
        raise ValueError(f"unknown arch: {arch}")
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    return m
