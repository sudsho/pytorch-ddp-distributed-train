"""Single-GPU baseline. Used as a sanity check before going to DDP."""
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import build_dataset
from src.model import build_model
from src.utils import load_config, setup_logger, Timer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = setup_logger("single")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")

    train_ds = build_dataset(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_per_gpu"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(cfg["model"]["arch"], cfg["data"]["num_classes"]).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"],
                    momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        with Timer() as t:
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad()
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
                running += loss.item()
            avg = running / max(1, len(train_loader))
        log.info(f"epoch {epoch} loss={avg:.4f} time={t.elapsed:.1f}s")


if __name__ == "__main__":
    main()
