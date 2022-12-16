# pytorch-ddp-distributed-train

Single-node multi-GPU training reference with PyTorch DDP. ResNet50 on Imagenette as the workload.

## What's in the repo

- Single-GPU baseline (`src/train_single.py`) for sanity checking before going parallel.
- DDP trainer (`src/train_ddp.py`) with `DistributedSampler` (and `set_epoch` for per-epoch reshuffling), AMP via `torch.cuda.amp` (autocast + `GradScaler`), gradient accumulation with `no_sync` on intermediate steps, optional `checkpoint_sequential` for memory-tight runs, cosine LR with warmup, float64 all-reduce of validation metrics, and rank-0-gated checkpointing.
- A second variant using HuggingFace `accelerate` (`src/train_accelerate.py`) for comparison. Note the accelerate variant does not currently run validation, save checkpoints, or log to MLflow, and its LR schedule is not identical to the DDP path.
- MLflow rank-0 logging of params and per-epoch metrics (DDP path only).
- Dockerfile based on `pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime`.

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
make test
```

## Results

No benchmark artifacts are committed in this repo. Training and benchmarking were done on a borrowed lab box that is not available to this checkout, and the raw sweep output directory is gitignored. Any throughput or accuracy numbers you want should be regenerated locally on your own hardware.

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
  utils.py            logger, config, mlflow hooks, timer
tests/          data, model, sched, smoke, checkpoint, utils
ci/             test.yml.example
scripts/        download_imagenette.sh
Dockerfile, Makefile
```

Tests cover the transform pipeline, model factory, scheduler shape, a checkpoint save/load roundtrip, a small utils suite, and a single CPU forward/backward smoke test. The DDP path itself is not exercised by the test suite.

## Notes I want to remember later

- The single biggest bug I hit: DDP + grad accumulation + `set_epoch` -- sampler shuffling is per-epoch, so if you forget `sampler.set_epoch(epoch)`, every rank sees the same shard order and you tank the training mix.
- `find_unused_parameters=True` is convenient and slow. Default it to False unless your forward really does skip params.
- `no_sync` during accum saves visible bandwidth but only matters above world=2 -- on 2 GPUs the all-reduce already overlaps fine.
- TODO: try ZeRO-1 via DeepSpeed when I get my hands on bigger cards. ResNet50 is too small to be a fair test for it though.

## License

MIT.
