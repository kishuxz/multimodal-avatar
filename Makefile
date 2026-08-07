.PHONY: setup provenance sweep analyze diffusion clean

setup:
	pip install -r requirements.txt

# Sanity-check the provenance helper on this host (GPU fields will be
# null off-GPU; that's expected, see bench/provenance.py).
provenance:
	python3 bench/provenance.py

# Phase 1: launches each arm's vLLM server and drives harness.py against
# it across the load/barge-in/prefix-cache grid. Added in Phase 1.
sweep:
	scripts/sweep.sh

# Phase 2: reads results/*.json, emits the markdown table and plots.
# Added in Phase 2.
analyze:
	python3 scripts/analyze.py

# Phase 3: per-frame diffusion timing breakdown. Added in Phase 3.
diffusion:
	python3 scripts/diffusion_bench.py

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
