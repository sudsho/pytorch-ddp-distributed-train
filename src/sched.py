"""LR schedulers.

Cosine with linear warmup, since that's what most ImageNet recipes use these days.
"""
import math


def build_scheduler(opt, cfg, steps_per_epoch):
    """steps_per_epoch should already account for grad_accum_steps.
    Caller is expected to pass len(loader) // accum, otherwise warmup ends way too early.
    """
    name = cfg["train"].get("scheduler", "none")
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 0)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup * steps_per_epoch)

    if name == "none":
        return None

    if name == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        import torch
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if name == "step":
        import torch
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1)

    raise ValueError(f"unknown scheduler {name}")
