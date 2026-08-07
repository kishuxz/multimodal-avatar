#!/bin/bash
# Phase 1 sweep driver. Runs on the pod, against the vLLM server it starts
# itself -- never invoked from a laptop against a remote host.
#
# Matrix:
#   arms:            fp16, awq (Qwen2.5-1.5B-Instruct-AWQ), fp8 (on-the-fly,
#                     --quantization fp8_per_tensor -- see docs/decisions.md
#                     for why this flag and not a bare "fp8")
#   load:             open-loop at rates derived per-arm (and per prefix-
#                     caching state, for fp16) via scripts/calibrate.py,
#                     targeting concurrency ~= 1 / 8 / 32
#   barge-in:         0.0 and 0.25
#   prefix caching:   off is the baseline for every arm (a large lever on
#                     this multi-turn, eight-conversation workload -- see
#                     docs/decisions.md for why leaving it on for one arm
#                     and not another would confound the comparison); fp16
#                     additionally runs with it on, as its own dimension
#   closed-loop:      one contrast run per arm, at the concurrency=8 rate,
#                     prefix caching off
#
# --gpu-memory-utilization is fixed at 0.9 for every server start in this
# script -- do not change it per-arm. See docs/decisions.md: it sets KV
# cache size, and letting it vary would make the comparison partly about
# the setting instead of the quantization.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_URL="http://localhost:8000/v1"
METRICS_URL="http://localhost:8000/metrics"
GPU_MEM_UTIL=0.9
CALIB_DURATION=30
LOAD_DURATION=120
CLOSED_LOOP_CONCURRENCY=8
BARGE_IN_FRACTIONS=(0.0 0.25)
TARGET_CONCURRENCIES=(1 8 32)
TMUX_SESSION="vllm-sweep"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

log() { echo "[sweep] $*"; }

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

confirm_cache_cold() {
    # A freshly started process has no prior state by construction, but
    # confirm rather than assume: kv_cache_usage_perc should read ~0
    # before any request has been served. (Verified against the running
    # server's actual /metrics output, not the metric name guessed from
    # memory -- vLLM 0.26.0 exposes this as vllm:kv_cache_usage_perc.)
    local usage
    usage=$(curl -s "$METRICS_URL" | grep '^vllm:kv_cache_usage_perc' | awk '{print $NF}')
    log "kv_cache_usage_perc after startup: ${usage:-unavailable}"
    if [ -n "$usage" ] && awk -v u="$usage" 'BEGIN{exit !(u > 0.01)}'; then
        echo "[sweep] cache usage is non-zero (${usage}) right after startup -- not cold, aborting" >&2
        exit 1
    fi
}

start_server() {
    # start_server <label> <model> [extra vllm flags...]
    local label="$1" model="$2"; shift 2

    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2

    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${used:-0}" -gt 500 ]; then
        echo "[sweep] GPU memory not freed after stopping previous server (${used} MiB) -- aborting" >&2
        exit 1
    fi

    log "starting server: label=$label model=$model extra_flags=[$*]"
    local logfile="/workspace/vllm_sweep_${label}.log"
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $REPO_ROOT && vllm serve $model --host 0.0.0.0 --port 8000 --gpu-memory-utilization $GPU_MEM_UTIL $* 2>&1 | tee $logfile"

    if ! wait_for_server_ready 300; then
        echo "[sweep] server did not become ready within 300s (label=$label) -- see $logfile" >&2
        exit 1
    fi
    confirm_cache_cold
    log "server ready: $label"
}

stop_server() {
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
}

run_calibration() {
    # run_calibration <model> <arm_label>
    local model="$1" arm_label="$2"
    local out="$RESULTS_DIR/calibration_${arm_label}.json"
    log "calibrating: $arm_label"
    python3 scripts/calibrate.py \
        --base-url "$BASE_URL" --model "$model" --label "$arm_label" \
        --duration "$CALIB_DURATION" \
        --target-concurrency "${TARGET_CONCURRENCIES[@]}" \
        --out "$out"
    echo "$out"
}

derived_rate() {
    # derived_rate <calibration_json> <target_concurrency>
    python3 -c "
import json
d = json.load(open('$1'))
print(d['derived_rates_req_per_s']['$2'])
"
}

run_open_loop_matrix() {
    # run_open_loop_matrix <arm_label> <model> <calibration_json>
    local arm_label="$1" model="$2" calib_json="$3"
    local conc rate barge_in out

    for conc in "${TARGET_CONCURRENCIES[@]}"; do
        rate=$(derived_rate "$calib_json" "$conc")
        for barge_in in "${BARGE_IN_FRACTIONS[@]}"; do
            out="$RESULTS_DIR/${arm_label}_open_c${conc}_bargein${barge_in}.json"
            log "open-loop: arm=$arm_label target_concurrency=$conc derived_rate=$rate barge_in=$barge_in"
            python3 harness.py \
                --base-url "$BASE_URL" --model "$model" \
                --mode open --rate "$rate" --duration "$LOAD_DURATION" \
                --barge-in "$barge_in" \
                --label "${arm_label}_c${conc}_bargein${barge_in}" \
                --out "$out"
        done
    done
}

run_closed_loop_contrast() {
    # run_closed_loop_contrast <arm_label> <model>
    local arm_label="$1" model="$2"
    local out="$RESULTS_DIR/${arm_label}_closed_c${CLOSED_LOOP_CONCURRENCY}.json"
    log "closed-loop contrast: arm=$arm_label concurrency=$CLOSED_LOOP_CONCURRENCY"
    python3 harness.py \
        --base-url "$BASE_URL" --model "$model" \
        --mode closed --concurrency "$CLOSED_LOOP_CONCURRENCY" --duration "$LOAD_DURATION" \
        --barge-in 0.0 \
        --label "${arm_label}_closed" \
        --out "$out"
}

run_arm() {
    # run_arm <arm_label> <model> <extra vllm flags...>
    #
    # Prefix caching off is the baseline for every arm -- it's a large
    # lever on this multi-turn, eight-conversation workload, and leaving
    # it on for one arm but not another would make a cross-arm latency
    # gap unattributable to the thing actually being compared
    # (quantization). fp16 additionally runs with it on, as its own
    # swept dimension, not as a difference baked into which arm gets
    # which cache state. See docs/decisions.md.
    local arm_label="$1" model="$2"; shift 2

    if [ "$arm_label" = "fp16" ]; then
        local pc_state
        for pc_state in off on; do
            local flag="--no-enable-prefix-caching"
            [ "$pc_state" = "on" ] && flag="--enable-prefix-caching"
            local label="${arm_label}_pc${pc_state}"

            start_server "$label" "$model" "$@" "$flag"
            local calib_json
            calib_json=$(run_calibration "$model" "$label")
            run_open_loop_matrix "$label" "$model" "$calib_json"
            if [ "$pc_state" = "off" ]; then
                run_closed_loop_contrast "$arm_label" "$model"
            fi
            stop_server
        done
    else
        start_server "$arm_label" "$model" "$@" "--no-enable-prefix-caching"
        local calib_json
        calib_json=$(run_calibration "$model" "$arm_label")
        run_open_loop_matrix "$arm_label" "$model" "$calib_json"
        run_closed_loop_contrast "$arm_label" "$model"
        stop_server
    fi
}

main() {
    log "sweep starting: results -> $RESULTS_DIR"

    run_arm fp16 "Qwen/Qwen2.5-1.5B-Instruct"
    run_arm awq  "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    run_arm fp8  "Qwen/Qwen2.5-1.5B-Instruct" --quantization fp8_per_tensor

    log "sweep complete"
}

main "$@"
