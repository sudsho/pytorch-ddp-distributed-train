# Benchmark notes

No benchmark output is committed in this repo. `benchmarks/raw/` is gitignored,
and any prior numbers I had were on a borrowed lab box that this checkout
cannot reproduce. Run `benchmarks/run_bench.sh` on your own hardware to
generate numbers that match your setup.

## Setup used previously (for reference only)

- Hardware: 4xV100 16GB.
- ImageNette train split, ~9.5k images, 224x224.
- ResNet50, SGD, batch_per_gpu=64.
- Warm up: 1 epoch ignored. Numbers averaged over the next 3 epochs.

Note the shipped sweep script routes world=1 through `src/train_single.py`
(no AMP, no imgs/sec log) and world=2,4 through `src/train_ddp.py`
(AMP + imgs/sec log). Those two code paths are not directly comparable, so
scaling-efficiency numbers derived from the sweep as-is would be misleading.
Fixing this would mean either adding AMP + throughput logging to
`train_single.py` or running the DDP script with `nproc=1` for the baseline.
