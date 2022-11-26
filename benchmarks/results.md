# Benchmark notes

## Setup

- Hardware: borrowed access to a 4xV100 16GB box from a friend's lab.
- ImageNette train split, ~9.5k images, 224x224.
- ResNet50, SGD, batch_per_gpu=64, AMP fp16.
- Warm up: 1 epoch ignored. Numbers averaged over the next 3 epochs.

## Throughput sweep

| world_size | batch_per_gpu | global_batch | imgs/sec | epoch_time |
|------------|---------------|--------------|----------|------------|
| 1          | 64            | 64           | 312      | 30.4 s     |
| 2          | 64            | 128          | 588      | 16.1 s     |
| 4          | 64            | 256          | 1148     | 8.3 s      |

## Scaling efficiency

ideal = N * single_gpu_throughput.

| world | imgs/sec | ideal | efficiency |
|-------|----------|-------|------------|
| 1     | 312      | 312   | 1.00       |
| 2     | 588      | 624   | 0.94       |
| 4     | 1148     | 1248  | 0.92       |

So ~92% efficiency at 4 GPUs on the lab box. Most of the lost 8% is the all-reduce
and the slightly less aggressive overlap of comm with compute on V100s.
With NCCL bucket size at default and `find_unused_parameters=False` it sits about here.

## With grad accumulation

global_batch=512, world=4, accum=2 (so per-step batch 128):

| amp | accum | imgs/sec |
|-----|-------|----------|
| on  | 1     | 1148     |
| on  | 2     | 1190     |
| off | 1     | 690      |

Accum gets a small boost because no_sync skips the all-reduce on the first step
of each pair. Not free though, the second step has 2x the gradients to reduce.

## Memory note

ResNet50 at bs=64 on V100 (16GB) with AMP uses about 9.5 GB activations.
With `grad_checkpoint=true`, drops to ~6.0 GB at the cost of ~12% slowdown.
Worth it only when scaling up batch or model size.
