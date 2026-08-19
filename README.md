# pytorch-ddp-distributed-train

Single-node multi-GPU training reference with PyTorch DDP. ResNet50 on Imagenette as the workload.

## Quick start (runs offline)

The full trainer (`src/train_ddp.py`) targets multi-GPU with the `nccl` backend and the Imagenette dataset, so it needs GPUs and a dataset download. To prove the DDP wiring without any of that, there is a CPU-only smoke that runs real DistributedDataParallel across 2 processes with the `gloo` backend on a tiny synthetic dataset. No CUDA, no download.

```bash
make smoke        # or: python scripts/smoke.py
```

Real output on a CPU box (torch 2.5.1, Python 3.11):

```
spawning 2 gloo processes on CPU ...
[rank 0/2] gloo up on 127.0.0.1:29529, backend=gloo, shard=32 samples
[rank 1/2] gloo up on 127.0.0.1:29529, backend=gloo, shard=32 samples
[rank 0] step  0 global_mean_loss=1.6909
[rank 1] step  0 global_mean_loss=1.6909
[rank 0] step 10 global_mean_loss=0.0010
[rank 0] step 20 global_mean_loss=0.5242
[rank 0] step 30 global_mean_loss=0.4696
[rank 0] step 39 global_mean_loss=0.0014
[rank 0] checkpoint saved -> checkpoints\smoke_epoch0.pt

==== DDP CPU smoke result ====
mode                : 2 processes (gloo)
backend             : gloo
loss (mean first3 -> last3 steps, rank-averaged): 1.2116 -> 0.0083 (DECREASED)
gradients all-reduced: spread across ranks = 0.00e+00 (SYNCED)
params in sync      : spread across ranks = 0.00e+00 (IN SYNC)
checkpoint (rank 0) : checkpoints\smoke_epoch0.pt (exists)

SMOKE PASSED
```

What the smoke proves: the loss decreases, gradients are all-reduced across ranks (the chosen gradient is bit-identical on every rank right after `backward`, so the spread is `0.00e+00`), model parameters stay in sync across ranks, and rank 0 (only) writes a checkpoint. The rank-averaged loss printed by both ranks is identical every step, which is itself a sign the ranks never diverge. A couple of harmless `socket.cpp ... failed to connect` warnings can appear on Windows while gloo resolves the loopback rendezvous; the all-reduce still succeeds.

If spawning 2 processes ever fails in your environment, the smoke automatically falls back to a single-process `gloo` group (`world_size=1`) that still exercises the DDP wrapper and `all_reduce`. That fallback cannot prove cross-rank gradient sync and says so. You can force it with `python scripts/smoke.py --procs 1`.

Run the tests (the DDP path is covered by `tests/test_ddp_smoke.py`, which shells out to the smoke):

```bash
make test         # or: python -m pytest -q tests/
# 13 passed
```

### What still needs real hardware

- **Multi-GPU / `nccl`**: `make ddp-2gpu` / `make ddp-4gpu` run `torchrun` with the `nccl` backend and `.cuda()` placement. They need actual GPUs on the box.
- **Multi-node**: `configs/multinode.yaml` and the `torchrun` rendezvous flags need a real multi-node cluster (`MASTER_ADDR` on the rank-0 host, matching `--nnodes` / `--node_rank`).
- **Dataset**: `make data` downloads Imagenette (~1.5 GB). The smoke sidesteps this with synthetic tensors.
- **AMP**: `torch.cuda.amp` autocast + `GradScaler` only kick in on CUDA; on CPU they are no-ops.

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
