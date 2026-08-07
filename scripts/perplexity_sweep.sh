#!/bin/bash
# Phase 4: perplexity on the fixed wikitext-2 slice, fp16 + AWQ. Runs on the
# pod, against a server it starts itself -- same shape as scripts/sweep.sh.
#
# FP8 is not run here. Issue #29: vLLM 0.19.1's online FP8 on this pod
# produces corrupted output from the first token, not degraded-but-coherent
# text -- a perplexity number from it would describe how confidently the
# model predicts token soup, not a quality comparison point. Re-add if #29
# is ever resolved.
#
# Prefix caching off, --gpu-memory-utilization 0.9 -- same baseline as the
# quantization latency sweep (docs/decisions.md), so this measurement sits
# on the same footing as the numbers it's being paired with in the writeup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_URL="http://localhost:8000/v1"
GPU_MEM_UTIL=0.9
REPEATS=5
TMUX_SESSION="vllm-sweep"
RESULTS_DIR="results/h200"
SLICE="data/wikitext2_test_slice_token_ids.json"

mkdir -p "$RESULTS_DIR"

log() { echo "[perplexity_sweep] $*" >&2; }

wait_for_server_ready() {
    local timeout_s="$1" waited=0
    while [ "$waited" -lt "$timeout_s" ]; do
        if curl -s -m 3 -o /dev/null -w '%{http_code}' "http://localhost:8000/v1/models" 2>/dev/null | grep -q '^200$'; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

start_server() {
    # start_server <label> <model> [extra vllm flags...]
    local label="$1" model="$2"; shift 2

    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2

    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${used:-0}" -gt 500 ]; then
        echo "[perplexity_sweep] GPU memory not freed after stopping previous server (${used} MiB) -- aborting" >&2
        exit 1
    fi

    log "starting server: label=$label model=$model extra_flags=[$*]"
    local logfile="/workspace/vllm_perplexity_${label}.log"
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $REPO_ROOT && HF_HOME=/workspace/hf vllm serve $model --host 0.0.0.0 --port 8000 --gpu-memory-utilization $GPU_MEM_UTIL --no-enable-prefix-caching $* 2>&1 | tee $logfile"

    if ! wait_for_server_ready 300; then
        echo "[perplexity_sweep] server did not become ready within 300s (label=$label) -- see $logfile" >&2
        exit 1
    fi

    cp "$logfile" "$RESULTS_DIR/server_log_perplexity_${label}.txt"
    log "server ready: $label (startup log -> $RESULTS_DIR/server_log_perplexity_${label}.txt)"
}

stop_server() {
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
}

run_arm() {
    # run_arm <label> <model> [extra vllm flags...]
    local label="$1" model="$2"; shift 2
    local out="$RESULTS_DIR/perplexity_${label}.json"

    start_server "$label" "$model" "$@"
    log "perplexity (${REPEATS} repeats): arm=$label"
    python3 scripts/perplexity.py \
        --base-url "$BASE_URL" --model "$model" --label "$label" \
        --slice "$SLICE" --repeats "$REPEATS" \
        --out "$out"
    stop_server
}

main() {
    log "perplexity sweep starting: results -> $RESULTS_DIR"

    run_arm fp16 "Qwen/Qwen2.5-1.5B-Instruct"
    run_arm awq  "Qwen/Qwen2.5-1.5B-Instruct-AWQ"

    log "perplexity sweep complete"
}

main
