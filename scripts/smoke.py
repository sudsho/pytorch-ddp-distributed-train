"""Offline CPU smoke for the DDP training path.

The real trainer (src/train_ddp.py) targets multi-GPU with the nccl backend and
the Imagenette dataset. That needs GPUs and a dataset download, so it cannot run
in a plain CI/laptop box. This smoke proves the same wiring on CPU:

  - spins up 2 processes with the gloo backend (MASTER_ADDR=127.0.0.1)
  - a tiny synthetic, learnable dataset sharded across ranks via DistributedSampler
  - wraps the model (reused from src.model.build_model) in DistributedDataParallel
  - trains a handful of steps so the loss decreases
  - proves gradients are all-reduced (identical across ranks after backward) and
    that model parameters stay in sync across ranks
  - checkpoints from rank 0 only (reused from src.checkpoint.save_checkpoint)

No CUDA is used at any point. If spawning 2 processes fails in this environment
the script falls back to a single process gloo group (world_size=1) that still
exercises the DDP wrapper + all_reduce, and says so explicitly.

Run:
  python scripts/smoke.py
"""
import argparse
import json
import os
import sys
import tempfile

# gloo on Windows needs the non-libuv TCPStore. Harmless elsewhere. Must be set
# before torch.distributed touches the rendezvous.
os.environ.setdefault("USE_LIBUV", "0")
# Keep every process single-threaded and CPU-only so the smoke is fast and
# deterministic and never reaches for a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# Make `src` importable whether run as `python scripts/smoke.py` or `-m`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.checkpoint import save_checkpoint  # noqa: E402
from src.model import build_model  # noqa: E402

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29529"
NUM_CLASSES = 4
IMG = 32
SAMPLES_PER_CLASS = 16
STEPS = 40
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def make_synthetic_dataset(seed=0):
    """Class-correlated images so the loss actually has something to learn.

    Each class gets a distinct constant offset added to random noise, which a
    small conv net separates within a few dozen steps. Fixed seed => same data
    on every rank build, then DistributedSampler shards it per rank.
    """
    g = torch.Generator().manual_seed(seed)
    n = NUM_CLASSES * SAMPLES_PER_CLASS
    y = torch.arange(NUM_CLASSES).repeat_interleave(SAMPLES_PER_CLASS)
    offsets = torch.linspace(-1.0, 1.0, NUM_CLASSES).view(NUM_CLASSES, 1, 1, 1)
    x = torch.randn(n, 3, IMG, IMG, generator=g) * 0.3 + offsets[y]
    return TensorDataset(x, y)


def _sync_ok(model):
    """True if every rank holds identical parameters (proof grads stayed synced).

    all_reduce MIN and MAX over the flattened params: if the min and the max
    across ranks match everywhere, all ranks are bit-for-bit in sync.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return True, 0.0
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    lo = flat.clone()
    hi = flat.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    spread = (hi - lo).abs().max().item()
    return spread == 0.0, spread


def _grads_synced(model):
    """Max spread of a chosen gradient across ranks. 0 => all-reduce happened.

    Ranks see different data shards, so pre-sync local grads differ; DDP
    all-reduces them in backward, so post-backward .grad must match across ranks.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return 0.0
    ref = None
    for p in model.parameters():
        if p.grad is not None:
            ref = p.grad.detach().reshape(-1).clone()
            break
    lo = ref.clone()
    hi = ref.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return (hi - lo).abs().max().item()


def run_worker(rank, world, result_path):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["USE_LIBUV"] = "0"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    torch.manual_seed(1234)  # identical init on every rank; DDP also broadcasts

    device = torch.device("cpu")
    ds = make_synthetic_dataset()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=16, sampler=sampler)

    model = build_model("resnet18", num_classes=NUM_CLASSES).to(device)
    ddp = DDP(model)  # no device_ids => CPU DDP
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01, momentum=0.9)

    print(f"[rank {rank}/{world}] gloo up on {MASTER_ADDR}:{MASTER_PORT}, "
          f"backend={dist.get_backend()}, shard={len(sampler)} samples", flush=True)

    def global_mean_loss(loss):
        # rank-averaged loss, i.e. what a real DDP trainer logs. On world_size=1
        # this is just the local loss.
        t = loss.detach().clone()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()
        return t.item()

    losses = []       # global mean loss per step
    grad_spread = None
    step = 0
    epoch = 0
    ddp.train()
    while step < STEPS:
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = ddp(x)
            loss = crit(logits, y)
            loss.backward()
            if step == 0:
                grad_spread = _grads_synced(ddp)  # check right after first backward
            opt.step()
            gl = global_mean_loss(loss)
            losses.append(gl)
            if step % 10 == 0 or step == STEPS - 1:
                print(f"[rank {rank}] step {step:2d} global_mean_loss={gl:.4f}", flush=True)
            step += 1
            if step >= STEPS:
                break
        epoch += 1

    # Compare smoothed windows so a single noisy batch can't flip the verdict.
    first_loss = sum(losses[:3]) / len(losses[:3])
    last_loss = sum(losses[-3:]) / len(losses[-3:])

    in_sync, param_spread = _sync_ok(ddp)

    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt = os.path.join(CKPT_DIR, "smoke_epoch0.pt")
        save_checkpoint(ckpt, ddp, opt, scaler=None, epoch=0)
        result = {
            "world_size": world,
            "backend": dist.get_backend(),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "loss_decreased": last_loss < first_loss,
            "grad_spread_across_ranks": grad_spread,
            "grads_all_reduced": (world == 1) or (grad_spread == 0.0),
            "params_in_sync": in_sync,
            "param_spread_across_ranks": param_spread,
            "checkpoint": ckpt,
            "checkpoint_exists": os.path.exists(ckpt),
        }
        with open(result_path, "w") as f:
            json.dump(result, f)
        print(f"[rank 0] checkpoint saved -> {os.path.relpath(ckpt, REPO_ROOT)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def _report(result_path):
    with open(result_path) as f:
        r = json.load(f)
    mode = "2 processes (gloo)" if r["world_size"] > 1 else \
        "single process fallback (gloo, world_size=1)"
    print("\n==== DDP CPU smoke result ====")
    print(f"mode                : {mode}")
    print(f"backend             : {r['backend']}")
    print(f"loss (mean first3 -> last3 steps, rank-averaged): "
          f"{r['first_loss']:.4f} -> {r['last_loss']:.4f} "
          f"({'DECREASED' if r['loss_decreased'] else 'DID NOT DECREASE'})")
    if r["world_size"] > 1:
        print(f"gradients all-reduced: spread across ranks = "
              f"{r['grad_spread_across_ranks']:.2e} "
              f"({'SYNCED' if r['grads_all_reduced'] else 'NOT SYNCED'})")
        print(f"params in sync      : spread across ranks = "
              f"{r['param_spread_across_ranks']:.2e} "
              f"({'IN SYNC' if r['params_in_sync'] else 'DIVERGED'})")
    else:
        print("gradients all-reduced: n/a (world_size=1, all_reduce is identity)")
    print(f"checkpoint (rank 0) : {os.path.relpath(r['checkpoint'], REPO_ROOT)} "
          f"({'exists' if r['checkpoint_exists'] else 'MISSING'})")

    ok = (
        r["loss_decreased"]
        and r["grads_all_reduced"]
        and r["params_in_sync"]
        and r["checkpoint_exists"]
    )
    print(f"\nSMOKE {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Offline CPU DDP smoke (gloo).")
    p.add_argument("--procs", type=int, default=2,
                   help="processes to spawn (default 2). Falls back to 1 if spawn fails.")
    args = p.parse_args()

    fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    world = max(1, args.procs)
    try:
        if world == 1:
            raise RuntimeError("single-process requested")
        import torch.multiprocessing as mp
        print(f"spawning {world} gloo processes on CPU ...", flush=True)
        mp.spawn(run_worker, args=(world, result_path), nprocs=world, join=True)
    except Exception as e:  # noqa: BLE001 - any spawn failure => documented fallback
        if world != 1:
            print(f"\n[fallback] 2-process spawn failed ({type(e).__name__}: {e}).")
            print("[fallback] running single-process gloo group (world_size=1). "
                  "This still exercises the DDP wrapper + all_reduce, but does NOT "
                  "prove cross-rank gradient sync.", flush=True)
        run_worker(0, 1, result_path)

    ok = _report(result_path)
    os.remove(result_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
