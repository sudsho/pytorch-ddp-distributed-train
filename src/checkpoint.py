"""Save and load training checkpoints. Rank 0 only saves."""
import os
import torch
import torch.distributed as dist


def save_checkpoint(path, model, opt, scaler, epoch, scheduler=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
    }
    torch.save(state, path)


def load_checkpoint(path, model, opt=None, scaler=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state["model"])
    if opt is not None and "opt" in state:
        opt.load_state_dict(state["opt"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0)
