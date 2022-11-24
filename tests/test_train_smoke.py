"""Single-step smoke test on a tiny synthetic dataset.

Skipped if CUDA isn't available since DDP would be a pain in CI without GPUs.
"""
import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model


def test_one_step_on_cpu():
    # tiny ResNet18 forward+backward on CPU just to make sure the loop wiring is sane.
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 10, (4,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)

    m = build_model("resnet18", num_classes=10)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)

    for xi, yi in loader:
        opt.zero_grad()
        out = m(xi)
        loss = crit(out, yi)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
        break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_amp_path():
    m = build_model("resnet18", num_classes=10).cuda()
    x = torch.randn(2, 3, 64, 64).cuda()
    y = torch.randint(0, 10, (2,)).cuda()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        out = m(x)
        loss = crit(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert torch.isfinite(loss)
