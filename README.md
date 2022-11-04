# pytorch-ddp-distributed-train

Distributed training experiments with PyTorch DDP. Multi-GPU on a single node and a stab at multi-node.

## Goal

Train a ResNet50 on Imagenette (10-class subset of ImageNet) with DDP and measure how throughput and accuracy scale as we add GPUs. Then write down what worked.

## Status

Just sketching the README. Code coming next.

## Plan

- ImageNette dataset loader (small enough to iterate fast)
- ResNet50 from torchvision
- single-GPU baseline first, then DDP
- AMP (mixed precision) for the throughput win
- benchmark across 1, 2, 4 GPUs
- write up scaling efficiency

## Why

I want to actually understand what `torch.distributed` does under the hood instead of copy-pasting boilerplate from a blog post. So I'll build it up piece by piece and break things on purpose to see the failure modes.
