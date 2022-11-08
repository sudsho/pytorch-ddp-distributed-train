"""Logging and small helpers."""
import logging
import os
import time
import yaml


def setup_logger(name="train", rank=0, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = f"[%(asctime)s][rank={rank}][%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class Timer:
    """Stupid simple wall-clock timer."""

    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
