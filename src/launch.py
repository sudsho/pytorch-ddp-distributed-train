"""Tiny wrapper around torchrun.

Just so I don't keep retyping the same command every 5 minutes.
"""
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("--standalone", action="store_true")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--master_addr", default="127.0.0.1")
    p.add_argument("--master_port", default="29500")
    args = p.parse_args()

    cmd = ["torchrun"]
    if args.standalone:
        cmd += ["--standalone"]
    cmd += [
        f"--nproc_per_node={args.nproc}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "-m", "src.train_ddp",
        "--config", args.config,
    ]
    print("running:", " ".join(cmd))
    env = os.environ.copy()
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
