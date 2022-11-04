# pytorch-ddp-distributed-train

Distributed training experiments with PyTorch DDP. Multi-GPU on a single node and a stab at multi-node.

## Goal

Train a ResNet50 on Imagenette (10-class subset of ImageNet) with DDP and measure how throughput and accuracy scale as we add GPUs.

## Dataset

[ImageNette](https://github.com/fastai/imagenette) is a 10-class subset of ImageNet that's small enough to iterate on locally without burning a hole in cloud bills. Classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute.

## Plan

- ImageNette dataset loader (small enough to iterate fast)
- ResNet50 from torchvision
- single-GPU baseline first, then DDP
- AMP (mixed precision) for the throughput win
- gradient accumulation for fitting bigger effective batch sizes
- benchmark across 1, 2, 4 GPUs
- multi-node with torchrun + rdzv backend
- write up scaling efficiency

## Layout (planned)

```
configs/   yaml configs
src/       data, model, train_ddp, train_accelerate, utils
benchmarks/  results + scripts
tests/     unit tests
notebooks/ profiling
```

## Why

Want to actually understand what `torch.distributed` does under the hood instead of copy-pasting boilerplate from a blog post. So building it up piece by piece and breaking things on purpose to see the failure modes.
