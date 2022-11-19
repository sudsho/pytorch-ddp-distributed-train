# Benchmark notes

## Setup

- Hardware: borrowed access to a 4xV100 16GB box from a friend's lab.
- ImageNette train split, ~9.5k images, 224x224.
- ResNet50, SGD, batch_per_gpu=64, AMP fp16.

## First run (single GPU)

torchrun --standalone --nproc_per_node=1, batch_per_gpu=64, AMP on.

Throughput: ~310 imgs/sec (averaged over 3 epochs after warm-up).
Epoch time: ~31s.

Will fill in the multi-GPU numbers as I get to them.
