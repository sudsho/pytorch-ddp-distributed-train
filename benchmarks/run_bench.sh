#!/usr/bin/env bash
# Sweep nproc_per_node = 1, 2, 4 and dump throughput numbers.
# Records to benchmarks/raw/<date>.txt so we can graph them later.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
DATESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="benchmarks/raw/${DATESTAMP}.txt"
mkdir -p benchmarks/raw

for N in 1 2 4; do
    echo "=== world=${N} ===" | tee -a "$OUT"
    if [ "$N" = "1" ]; then
        python -m src.train_single --config "$CONFIG" 2>&1 | tee -a "$OUT"
    else
        torchrun --standalone --nproc_per_node="$N" -m src.train_ddp --config "$CONFIG" 2>&1 | tee -a "$OUT"
    fi
done

echo "raw output: $OUT"
