"""Sanity checks on the model factory."""
import torch
from src.model import build_model


def test_resnet18_shape():
    m = build_model("resnet18", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 10)


def test_resnet50_shape():
    m = build_model("resnet50", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 10)


def test_grad_checkpoint_forward():
    m = build_model("resnet18", num_classes=10, grad_checkpoint=True)
    m.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 10)
    y.sum().backward()
