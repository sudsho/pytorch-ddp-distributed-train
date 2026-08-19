"""Runs the offline CPU DDP smoke and asserts it reports success.

This is the only test that exercises the actual DistributedDataParallel path
(2 gloo processes on CPU with cross-rank gradient all-reduce). It shells out to
scripts/smoke.py so the real multiprocessing spawn is covered end to end. No
CUDA, no dataset download.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(REPO_ROOT, "scripts", "smoke.py")


def test_ddp_cpu_smoke_passes():
    proc = subprocess.run(
        [sys.executable, SMOKE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout + proc.stderr
    assert "SMOKE PASSED" in out, f"smoke did not pass:\n{out}"
    assert proc.returncode == 0, f"nonzero exit {proc.returncode}:\n{out}"
    # the run must have taken the real 2-process path, not the fallback
    assert "2 processes (gloo)" in out, f"expected 2-proc gloo run:\n{out}"
    assert "DECREASED" in out
    assert "SYNCED" in out  # gradients all-reduced across ranks
