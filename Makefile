PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: help install data single ddp-2gpu ddp-4gpu accelerate bench test clean

help:
	@echo "single        run single-gpu baseline"
	@echo "ddp-2gpu      torchrun on 2 procs (single node)"
	@echo "ddp-4gpu      torchrun on 4 procs (single node)"
	@echo "accelerate    accelerate launch variant"
	@echo "bench         throughput benchmark across sizes"
	@echo "test          pytest"

install:
	pip install -r requirements.txt

data:
	bash scripts/download_imagenette.sh

single:
	$(PYTHON) -m src.train_single --config $(CONFIG)

ddp-2gpu:
	torchrun --standalone --nproc_per_node=2 -m src.train_ddp --config $(CONFIG)

ddp-4gpu:
	torchrun --standalone --nproc_per_node=4 -m src.train_ddp --config $(CONFIG)

accelerate:
	accelerate launch -m src.train_accelerate --config $(CONFIG)

bench:
	bash benchmarks/run_bench.sh

test:
	pytest -q tests/

clean:
	rm -rf checkpoints/ profiles/ mlruns/
