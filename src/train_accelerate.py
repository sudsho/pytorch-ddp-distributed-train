"""Same training, but with HuggingFace accelerate doing the boilerplate.

Mostly here for comparison: how much glue do we save vs raw torchrun + DDP?

  accelerate config        # one-time
  accelerate launch -m src.train_accelerate --config configs/default.yaml
"""
import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.data import build_dataset
from src.model import build_model
from src.sched import build_scheduler
from src.utils import load_config, setup_logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.get("amp", {}).get("enabled", False) else "no",
        gradient_accumulation_steps=cfg["train"].get("grad_accum_steps", 1),
    )
    log = setup_logger("accel", rank=accelerator.process_index)
    log.info(f"world={accelerator.num_processes}")

    train_ds = build_dataset(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    val_ds = build_dataset(cfg["data"]["root"], "val", cfg["data"]["image_size"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_per_gpu"],
        shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_per_gpu"],
        shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    model = build_model(
        cfg["model"]["arch"], cfg["data"]["num_classes"],
        grad_checkpoint=cfg["model"].get("grad_checkpoint", False),
    )
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = build_scheduler(opt, cfg, len(train_loader))

    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)
    crit = nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in train_loader:
            with accelerator.accumulate(model):
                opt.zero_grad()
                logits = model(x)
                loss = crit(logits, y)
                accelerator.backward(loss)
                opt.step()
                if sched is not None:
                    sched.step()
        if accelerator.is_main_process:
            log.info(f"epoch {epoch} done")


if __name__ == "__main__":
    main()
