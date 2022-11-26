"""Tests for the small helpers."""
import time
import pytest

from src.utils import Timer, load_config


def test_timer_records_elapsed():
    t = Timer()
    with t:
        time.sleep(0.05)
    assert t.elapsed >= 0.045  # allow some clock drift, don't be too tight or it flakes
    assert t.elapsed < 1.0


def test_load_config_default(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(str(p))
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2
