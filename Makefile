.PHONY: setup provenance sweep analyze analyze-h200 kv-check perplexity perplexity-slices diffusion diffusion-quality clean

setup:
	pip install -r requirements.txt

# Sanity-check the provenance helper on this host (GPU fields will be
# null off-GPU; that's expected, see bench/provenance.py).
provenance:
	python3 bench/provenance.py

# H200 sweep driver: launches each arm's vLLM server and drives harness.py
# against it across the load/barge-in/prefix-cache grid, writing to
# results/h200/. This is the only sweep target current scripts can run --
# the H100 pod that produced results/*.json (flat) was reclaimed; those
# results are kept as a static, non-reproducible prior run. See README,
# "Environment and the H100 prior run."
sweep:
	scripts/sweep.sh

# Reads results/*.json (H100), emits the markdown table and plots. GPU-free
# -- runs against committed JSON only, no server.
analyze:
	python3 scripts/analyze.py

# Same as `analyze`, pointed at the H200 results instead. Kept as a
# separate target rather than a flag on `analyze` so both are one command
# each -- analyze.py itself takes --results-dir/--plots-dir/--out if a
# different split is ever needed.
analyze-h200:
	python3 scripts/analyze.py --results-dir results/h200 --plots-dir plots/h200 --out results/h200/summary.md

# Expected KV cache bytes/token and bytes/sequence from the model config,
# arithmetic only -- no server, no GPU.
kv-check:
	python3 scripts/kv_cache_check.py

# Perplexity across 8 distinct wikitext-2 slices, fp16 + AWQ, against a
# server this script starts itself. Requires a GPU pod, not runnable from
# a laptop.
perplexity:
	scripts/perplexity_sweep.sh

# One-time, offline, no GPU: cuts the 8 wikitext-2 slices `perplexity`
# scores. Already run once; data/wikitext2_test_slices* is committed, so
# this only needs re-running if the slice count/length ever changes.
# Needs pyarrow/transformers/huggingface_hub locally -- not in
# requirements.txt, see docs/decisions.md.
perplexity-slices:
	python3 scripts/build_perplexity_slices.py

# Phase 6: per-frame stage timing (conditioning/step/VAE decode) across
# step counts 1-20, with and without DeepCache, at 512x512. Requires a
# GPU; no vLLM, no server -- just requirements-pod-diffusion.txt on top
# of a normal CUDA/torch install.
diffusion:
	scripts/diffusion_sweep.sh

# LPIPS distance between a DeepCache frame and the same seed/steps
# without caching -- run after `diffusion`, once the matched-seed image
# pairs it produces exist.
diffusion-quality:
	python3 scripts/diffusion_quality_check.py \
		--reference results/h200/diffusion/quality_steps20_nocache.png \
		--candidate results/h200/diffusion/quality_steps20_deepcache.png \
		--out results/h200/diffusion/quality_steps20_lpips.json

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
