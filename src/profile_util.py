"""Helpers around torch.profiler for short profiling runs."""
from contextlib import contextmanager
import os
import torch


@contextmanager
def profile_steps(out_dir="profiles", wait=1, warmup=1, active=3, repeat=1):
    os.makedirs(out_dir, exist_ok=True)
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    handler = torch.profiler.tensorboard_trace_handler(out_dir)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    )
    prof.start()
    try:
        yield prof
    finally:
        prof.stop()
