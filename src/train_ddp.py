"""DDP trainer.

Launch with torchrun:
  torchrun --standalone --nproc_per_node=2 -m src.train_ddp --config configs/default.yaml
"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data import build_dataset
from src.eval import validate
from src.model import build_model
from src.utils import load_config, setup_logger


def setup_dist(backend="nccl"):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    rank, world, local_rank = setup_dist(cfg["ddp"]["backend"])
    log = setup_logger("ddp", rank=rank)
    log.info(f"world={world} local_rank={local_rank}")

    train_ds = build_dataset(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    val_ds = build_dataset(cfg["data"]["root"], "val", cfg["data"]["image_size"])
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_per_gpu"],
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_per_gpu"],
        sampler=val_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(cfg["model"]["arch"], cfg["data"]["num_classes"]).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for x, y in loader:
            x = x.cuda(local_rank, non_blocking=True)
            y = y.cuda(local_rank, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
        val_loss, val_acc = validate(model, val_loader, torch.device(f"cuda:{local_rank}"), distributed=True)
        if rank == 0:
            log.info(
                f"epoch {epoch} avg_loss={running/max(1,len(loader)):.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    cleanup_dist()


if __name__ == "__main__":
    main()
