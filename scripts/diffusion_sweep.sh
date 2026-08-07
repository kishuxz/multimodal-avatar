#!/bin/bash
# Phase 6: diffusion frame-budget sweep. Runs on the pod (or any CUDA
# machine with the deps below) -- no server, no vLLM, plain Python
# processes. Two passes over the same step-count list: without
# DeepCache (the stage-cost / ceiling question) and with it (the
# optimization question) -- see docs/decisions.md for the step-count
# choices and cache hyperparameters.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="results/h200/diffusion"
REPEATS=5
# Dense near the 40-100ms boundary (where the real question is), sparser
# above it (context for the DeepCache comparison at a normal "quality"
# step count).
STEP_COUNTS=(1 2 3 4 5 8 12 20)
CACHE_INTERVAL=5
CACHE_BRANCH=0

mkdir -p "$RESULTS_DIR"

log() { echo "[diffusion_sweep] $*" >&2; }

for n in "${STEP_COUNTS[@]}"; do
    log "no-cache: steps=$n"
    python3 scripts/diffusion_bench.py --steps "$n" --repeats "$REPEATS" \
        --out "$RESULTS_DIR/steps${n}.json"

    log "deep-cache: steps=$n interval=$CACHE_INTERVAL branch=$CACHE_BRANCH"
    python3 scripts/diffusion_bench.py --steps "$n" --repeats "$REPEATS" \
        --deep-cache --cache-interval "$CACHE_INTERVAL" --cache-branch "$CACHE_BRANCH" \
        --out "$RESULTS_DIR/steps${n}_deepcache.json"
done

# Matched-seed quality comparison images, saved alongside the JSON --
# one low step count near the real budget ceiling, one typical
# "quality" step count, so the DeepCache quality cost can be judged at
# both ends, not just the flattering one.
for n in 4 20; do
    log "quality-comparison images: steps=$n"
    python3 scripts/diffusion_bench.py --steps "$n" --repeats 1 --seed-start 0 \
        --out "$RESULTS_DIR/quality_steps${n}_nocache.json" \
        --save-image "$RESULTS_DIR/quality_steps${n}_nocache.png"
    python3 scripts/diffusion_bench.py --steps "$n" --repeats 1 --seed-start 0 \
        --deep-cache --cache-interval "$CACHE_INTERVAL" --cache-branch "$CACHE_BRANCH" \
        --out "$RESULTS_DIR/quality_steps${n}_deepcache.json" \
        --save-image "$RESULTS_DIR/quality_steps${n}_deepcache.png"
done

log "diffusion sweep complete"
