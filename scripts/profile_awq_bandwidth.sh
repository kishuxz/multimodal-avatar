#!/bin/bash
# Phase 8: profile fp16 and AWQ under real, sustained c~=32 open-loop
# load with torch.profiler enabled -- same server-start shape as
# scripts/sweep.sh, plus --profiler-config so vLLM's own
# /start_profile and /stop_profile endpoints work.
#
# Rates below are the existing calibration for c~=32
# (results/h200/calibration_{fp16_pcoff,awq}.json) -- not
# re-calibrated here, that measurement is already valid and unrelated
# to what's being profiled.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_URL="http://localhost:8000/v1"
GPU_MEM_UTIL=0.9
TMUX_SESSION="vllm-profile"
RESULTS_DIR="results/h200"
PROFILE_ROOT="/workspace/profiles"

FP16_RATE=475.0453
AWQ_RATE=431.3068

mkdir -p "$RESULTS_DIR"

log() { echo "[profile_awq_bandwidth] $*" >&2; }

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
    # start_server <label> <model> <profiler_dir> [extra vllm flags...]
    local label="$1" model="$2" profiler_dir="$3"; shift 3

    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2

    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${used:-0}" -gt 500 ]; then
        echo "[profile_awq_bandwidth] GPU memory not freed after stopping previous server (${used} MiB) -- aborting" >&2
        exit 1
    fi

    mkdir -p "$profiler_dir"
    log "starting server: label=$label model=$model profiler_dir=$profiler_dir"
    local logfile="/workspace/vllm_profile_${label}.log"
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $REPO_ROOT && HF_HOME=/workspace/hf vllm serve $model --host 0.0.0.0 --port 8000 --gpu-memory-utilization $GPU_MEM_UTIL --no-enable-prefix-caching --profiler-config.profiler torch --profiler-config.torch_profiler_dir $profiler_dir --profiler-config.torch_profiler_with_stack false $* 2>&1 | tee $logfile"

    if ! wait_for_server_ready 300; then
        echo "[profile_awq_bandwidth] server did not become ready within 300s (label=$label) -- see $logfile" >&2
        exit 1
    fi
    cp "$logfile" "$RESULTS_DIR/server_log_profile_${label}.txt"
    log "server ready: $label"
}

stop_server() {
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
}

run_arm() {
    # run_arm <label> <model> <rate>
    local label="$1" model="$2" rate="$3"
    local profiler_dir="$PROFILE_ROOT/$label"
    local out="$RESULTS_DIR/profile_awq_bandwidth_${label}.json"

    start_server "$label" "$model" "$profiler_dir"
    log "profiling: arm=$label rate=$rate"
    python3 scripts/profile_awq_bandwidth.py \
        --base-url "$BASE_URL" --model "$model" --label "$label" --rate "$rate" \
        --profiler-dir "$profiler_dir" --out "$out"
    stop_server
}

main() {
    log "AWQ bandwidth profiling starting -> $RESULTS_DIR"

    run_arm fp16 "Qwen/Qwen2.5-1.5B-Instruct" "$FP16_RATE"
    run_arm awq  "Qwen/Qwen2.5-1.5B-Instruct-AWQ" "$AWQ_RATE"

    log "profiling complete"
}

main
