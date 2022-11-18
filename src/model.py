"""Model factory. Currently just torchvision resnets."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import models


class CheckpointedResNet(nn.Module):
    """ResNet wrapper that runs the body through checkpoint_sequential.

    Saves a chunk of activations at the cost of an extra forward pass.
    """

    def __init__(self, base, segments=4):
        super().__init__()
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.body = nn.Sequential(base.layer1, base.layer2, base.layer3, base.layer4)
        self.head = nn.Sequential(base.avgpool, nn.Flatten(1), base.fc)
        self.segments = segments

    def forward(self, x):
        x = self.stem(x)
        if self.training:
            x = checkpoint_sequential(self.body, self.segments, x)
        else:
            x = self.body(x)
        return self.head(x)


def build_model(arch="resnet50", num_classes=10, pretrained=False, grad_checkpoint=False):
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
    if grad_checkpoint:
        m = CheckpointedResNet(m)
    return m
