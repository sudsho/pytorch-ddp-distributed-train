"""Round-trip checkpoint save/load."""
import os
import torch

from src.checkpoint import save_checkpoint, load_checkpoint
from src.model import build_model


def test_save_load_roundtrip(tmp_path):
    m = build_model("resnet18", num_classes=10)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # before-state: snapshot a parameter
    p_before = next(m.parameters()).detach().clone()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, m, opt, scaler, epoch=2)
    assert os.path.exists(path)

    # mutate, then load and check we recover the original.
    with torch.no_grad():
        next(m.parameters()).fill_(0.0)
    epoch = load_checkpoint(path, m, opt, scaler)
    assert epoch == 2
    p_after = next(m.parameters()).detach().clone()
    assert torch.allclose(p_before, p_after)
