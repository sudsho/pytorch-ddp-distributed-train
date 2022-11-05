"""Model factory. Currently just torchvision resnets."""
from torchvision import models


def build_model(arch="resnet50", num_classes=10, pretrained=False):
    if arch == "resnet50":
        m = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    elif arch == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    else:
        raise ValueError(f"unknown arch: {arch}")
    in_feats = m.fc.in_features
    m.fc = __import__("torch").nn.Linear(in_feats, num_classes)
    return m
