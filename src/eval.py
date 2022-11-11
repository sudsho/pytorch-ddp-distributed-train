"""Validation pass with all-reduce of metrics."""
import torch
import torch.distributed as dist


@torch.no_grad()
def validate(model, loader, device, distributed=False):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = torch.nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    if distributed and dist.is_initialized():
        # all-reduce sums across ranks
        t = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum, correct, total = t.tolist()

    return loss_sum / max(1, total), correct / max(1, total)
