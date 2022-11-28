# pytorch-ddp-distributed-train

Distributed training with PyTorch DDP. Multi-GPU on a single node, plus a working multi-node setup via `torchrun` rdzv. Throughput and scaling efficiency measured on a 4xV100 box. ResNet50 trained on Imagenette as the workload.

## What's in the box

- Single-GPU baseline (`src/train_single.py`) for sanity checking before going parallel.
- DDP trainer (`src/train_ddp.py`) with `DistributedSampler`, AMP via `torch.cuda.amp`, gradient accumulation with `no_sync` on intermediate steps, optional `checkpoint_sequential` for memory-tight runs, cosine LR with warmup.
- Same loop with HuggingFace `accelerate` (`src/train_accelerate.py`) for comparison.
- Profiler hook (`src/profile_util.py`) writing tensorboard traces.
- MLflow rank-0 logging of params + per-epoch metrics.
- Benchmark sweep (`benchmarks/run_bench.sh`) across 1, 2, 4 GPUs, with results in `benchmarks/results.md`.
- Docker image based on the official `pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime` plus a 2-node compose file for testing rdzv on one machine.

## Dataset

[Imagenette](https://github.com/fastai/imagenette) is a 10-class subset of ImageNet (320px shortest side variant). Smaller than full ImageNet so iteration is fast.

```
make data    # downloads imagenette2-320 to ./data
```

Classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute.

## Quickstart

```bash
make install
make data
make single                  # single-gpu baseline
make ddp-2gpu                # torchrun on 2 procs
make ddp-4gpu                # torchrun on 4 procs
make accelerate              # accelerate launch variant
make bench                   # 1/2/4 sweep, dumps to benchmarks/raw/
make test
```

Multi-node (per node):

```bash
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.1:29500 \
  -m src.train_ddp --config configs/multinode.yaml
```

## Results (4xV100, ResNet50, AMP fp16, batch_per_gpu=64)

| world_size | imgs/sec | scaling efficiency |
|-----------:|---------:|-------------------:|
| 1          | 312      | 1.00               |
| 2          | 588      | 0.94               |
| 4          | 1148     | 0.92               |

Full table in `benchmarks/results.md`.

Top-1 val accuracy on Imagenette after 5 epochs of training from scratch with cosine LR + warmup: ~78%. Not the point of this repo (the point is the systems plumbing) but good enough that I know the loop is correct.

## Layout

```
configs/        default.yaml + multinode.yaml
src/
  data.py             imagenette transforms + ImageFolder loader
  model.py            resnet factory + checkpoint_sequential wrapper
  train_single.py     single-gpu baseline
  train_ddp.py        torchrun + DDP, AMP, accum, MLflow
  train_accelerate.py accelerate variant
  launch.py           thin torchrun wrapper
  sched.py            cosine + warmup
  eval.py             validation with all-reduce of metrics
  checkpoint.py       save/load (rank-0 only saves)
  profile_util.py     torch.profiler context
  utils.py            logger, config, mlflow hooks, timer
benchmarks/     run_bench.sh, results.md
notebooks/      profile.ipynb
tests/          data, model, sched, smoke, checkpoint, utils
ci/             test.yml.example (move to .github/workflows/ when ready)
scripts/        download_imagenette.sh
Dockerfile, docker-compose.yml, Makefile
```

## Notes I want to remember later

- The single biggest bug I hit: DDP + grad accumulation + `set_epoch` -- sampler shuffling is per-epoch, so if you forget `sampler.set_epoch(epoch)`, every rank sees the same shard order and you tank the training mix.
- `find_unused_parameters=True` is convenient and slow. Default it to False unless your forward really does skip params.
- AMP is essentially free on V100; it stops being free on consumer cards because their tensor cores are smaller.
- `no_sync` during accum saves visible bandwidth but only matters above world=2 -- on 2 GPUs the all-reduce already overlaps fine.
- TODO: try ZeRO-1 via DeepSpeed when I get my hands on bigger cards. ResNet50 is too small to be a fair test for it though.

## License

MIT.
