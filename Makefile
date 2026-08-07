.PHONY: setup provenance sweep analyze kv-check perplexity diffusion clean

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

# Phase 5: reads results/*.json, emits the markdown table and plots.
# GPU-free -- runs against committed JSON only, no server. Added in Phase 5
# (this comment previously said Phase 2, from before analyze.py existed).
analyze:
	python3 scripts/analyze.py

# Phase 5: expected KV cache bytes/token and bytes/sequence from the model
# config, arithmetic only -- no server, no GPU. Added in Phase 5.
kv-check:
	python3 scripts/kv_cache_check.py

# Phase 4: perplexity across 8 distinct wikitext-2 slices, fp16 + AWQ,
# against a server this script starts itself. Requires a GPU pod, not
# runnable from a laptop. The slices themselves (data/wikitext2_test_slices*)
# are prepared separately, offline, by scripts/build_perplexity_slices.py --
# not part of this target, since it needs pyarrow/transformers/
# huggingface_hub locally and only ever needs to run once.
perplexity:
	scripts/perplexity_sweep.sh

# Phase 3: per-frame diffusion timing breakdown. Added in Phase 3.
diffusion:
	python3 scripts/diffusion_bench.py

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
