import torch

from src.sched import build_scheduler


def _opt(lr=0.1):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_cosine_warmup_starts_at_zero():
    cfg = {"train": {"scheduler": "cosine", "epochs": 5, "warmup_epochs": 1, "lr": 0.1}}
    opt = _opt(lr=0.1)
    sch = build_scheduler(opt, cfg, steps_per_epoch=10)
    # at step 0, multiplier should be 0 (warmup_steps=10, current=0), so linear
    # warmup starts the lr at zero.
    assert sch is not None
    assert opt.param_groups[0]["lr"] == 0.0
    # after one step, lr is some positive fraction of base
    sch.step()
    assert 0 < opt.param_groups[0]["lr"] <= 0.1


def test_none_returns_none():
    cfg = {"train": {"scheduler": "none", "epochs": 1, "warmup_epochs": 0, "lr": 0.1}}
    opt = _opt()
    assert build_scheduler(opt, cfg, 10) is None
