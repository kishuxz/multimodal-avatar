#!/bin/bash
# Sweep driver, v2 -- see docs/decisions.md for what changed from the
# original Phase 1-3 (H100) version and why. Runs on the pod, against the
# vLLM server it starts itself -- never invoked from a laptop against a
# remote host.
#
# Matrix:
#   arms:            fp16, awq (Qwen2.5-1.5B-Instruct-AWQ), fp8 (on-the-fly
#                     --quantization <flag> -- see docs/decisions.md for
#                     which flag string is correct for the vLLM version
#                     actually installed; it has changed at least once)
#   load:             open-loop at rates derived per-arm (and per prefix-
#                     caching state, for fp16) via scripts/calibrate.py,
#                     targeting concurrency ~= 1 / 8 / 32
#   barge-in:         0.0 and 0.25, abort delay scaled per arm -- see
#                     BARGE_IN_MIN_FRAC/BARGE_IN_MAX_FRAC below
#   prefix caching:   off is the baseline for every arm (a large lever on
#                     this multi-turn, eight-conversation workload -- see
#                     docs/decisions.md for why leaving it on for one arm
#                     and not another would confound the comparison); fp16
#                     additionally runs with it on, as its own dimension
#   closed-loop:      one contrast run per arm, at the concurrency=8 rate,
#                     prefix caching off
#   repeats:          5 seeds at concurrency ~= 1 and ~= 32, prefix caching
#                     off, barge-in 0.0 -- the cells the quantization
#                     comparison actually hangs on. Built into this script
#                     (scripts/repeat_check.py, invoked inline) rather than
#                     a separate manual pass after the fact.
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
# 20s, not the original 120s: validated via a duration-stability check
# at the concurrency=32 rate (the expensive one -- ~500 req/s means a
# 120s run is 60,000+ requests) -- 20s vs 60s gave TTFT p99 106.9ms vs
# 99.5ms (n=7970 vs n=23831), ~7% apart, judged stable enough. Applied
# uniformly, which is a real tradeoff at the low end: concurrency~=1
# (~16 req/s) gets ~300 samples in 20s versus ~1900 at 120s -- p50 is
# still fine there, p99 is noisier than the high-concurrency points'.
# See docs/decisions.md.
CALIB_DURATION=30
LOAD_DURATION=20
CLOSED_LOOP_CONCURRENCY=8
BARGE_IN_FRACTIONS=(0.0 0.25)
TARGET_CONCURRENCIES=(1 8 32)
# Scaled barge-in window (v2): the original design sampled the abort
# delay from a FIXED 0.3-1.2s range regardless of arm or load. Every
# arm's response finished well inside that range at concurrency ~= 1 and
# ~= 8 (mean service time on the order of tens to ~150ms there), so the
# abort timer essentially never fired before the request completed on
# its own -- confirmed directly in the H100 data: aborted_total was 0 at
# every c1/c8 bargein0.25 run, only c32 (where queueing pushed response
# time up near the fixed window) ever produced an abort. That meant the
# barge-in dimension of the whole sweep was only ever tested at one of
# its three load points. Scaling the window to a fraction of each arm's
# OWN calibrated mean service time (from the concurrency=1 calibration
# probe, scripts/calibrate.py) fixes that: the window is always sized
# relative to how long that arm's responses actually take, so it lands
# inside a still-in-flight request at all three load points instead of
# only the slowest one. Chosen fractions -- 25% to 75% of mean service
# time -- center the window mid-response: early enough to reliably clear
# TTFT (a small fraction of total service time for these arms) without
# firing before generation starts, late enough to still leave a real gap
# before natural completion.
#
# Known limitation, stated rather than hidden: "mean service time" is
# measured unqueued (concurrency=1). At concurrency ~= 32, actual
# response time is longer than that due to queueing, so a window sized
# off the low-load figure can land earlier in the ACTUAL (queued)
# request lifetime than "mid-decode" -- it will still reliably fire and
# interrupt something in flight (the bug being fixed), just not
# necessarily at the same relative point in the response every time.
# Holding the window to a single per-arm scale, not one recalibrated per
# load point, is a deliberate simplification, not an oversight -- see
# docs/decisions.md.
BARGE_IN_MIN_FRAC=0.25
BARGE_IN_MAX_FRAC=0.75
# Repeats built into the matrix (v2): the cells the quantization
# comparison's noise band actually depends on. Concurrency 8 and
# barge-in 0.25 are deliberately not repeated -- same scope
# repeat_check.py's original standalone usage had (see docs/decisions.md,
# "Corrected results").
REPEAT_CONCURRENCIES=(1 32)
REPEATS=5
TMUX_SESSION="vllm-sweep"
# results/h200, not results/ -- this run's filenames (fp16_pcoff_open_c1_
# bargein0.0.json, calibration_fp16_pcoff.json, ...) are identical to the
# H100 run's, which are already committed flat under results/. Writing
# there would silently overwrite the H100 baseline this repo just spent
# an entry (docs/decisions.md, "The H200 run is a fresh baseline") saying
# never gets mixed or replaced. A subdirectory boundary makes that
# structural, not a naming convention someone could get wrong.
RESULTS_DIR="results/h200"

mkdir -p "$RESULTS_DIR"

# stderr, not stdout: several functions below (run_calibration,
# derived_rate) return a value via `echo` for the caller to capture with
# $(...), which grabs everything the function prints to stdout -- a log
# line here would silently corrupt that capture. (Caught exactly this
# way: log() on stdout mixed into calib_json, corrupting the path with
# log text, which broke derived_rate()'s embedded Python with a
# SyntaxError several steps later. The whole point of `tee`-ing this
# script's own stdout+stderr together in the caller is that nothing is
# lost by moving log() here.)
log() { echo "[sweep] $*" >&2; }

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
    # memory -- confirmed present under this name on both vLLM 0.26.0
    # and 0.19.1.)
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
    # HF_HOME is set explicitly here rather than relied on from ~/.bashrc --
    # whether tmux's shell sources it depends on login-shell behavior this
    # script shouldn't have to assume.
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $REPO_ROOT && HF_HOME=/workspace/hf vllm serve $model --host 0.0.0.0 --port 8000 --gpu-memory-utilization $GPU_MEM_UTIL $* 2>&1 | tee $logfile"

    if ! wait_for_server_ready 300; then
        echo "[sweep] server did not become ready within 300s (label=$label) -- see $logfile" >&2
        exit 1
    fi
    confirm_cache_cold

    # Server startup log, committed (v2, closes issue #19). Copied here,
    # right after the server is confirmed ready and before any load-test
    # traffic has been sent, so this is a clean startup-only snapshot --
    # resolved config, GPU KV cache size, max model length, block size,
    # attention backend selection -- without needing to grep it out of a
    # log that later also carries request-serving output. Previously this
    # only ever lived at $logfile, on the pod's local disk, never
    # committed -- the KV-cache arithmetic check
    # (scripts/kv_cache_check.py) had nothing to verify against as a
    # direct result.
    cp "$logfile" "$RESULTS_DIR/server_log_${label}.txt"
    log "server ready: $label (startup log -> $RESULTS_DIR/server_log_${label}.txt)"
}

stop_server() {
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
}

run_calibration() {
    # run_calibration <model> <arm_label>
    # Only `echo "$out"` at the end may go to this function's stdout --
    # the caller captures it with $(...). calibrate.py's own printed
    # summary is useful but goes to stderr here so it can't contaminate
    # that capture the same way log() almost did.
    local model="$1" arm_label="$2"
    local out="$RESULTS_DIR/calibration_${arm_label}.json"
    log "calibrating: $arm_label"
    python3 scripts/calibrate.py \
        --base-url "$BASE_URL" --model "$model" --label "$arm_label" \
        --duration "$CALIB_DURATION" \
        --target-concurrency "${TARGET_CONCURRENCIES[@]}" \
        --out "$out" >&2
    echo "$out"
}

derived_rate() {
    # derived_rate <calibration_json> <target_concurrency>
    # calibrate.py parses --target-concurrency with type=float, so its
    # derived_rates_req_per_s keys are "1.0"/"8.0"/"32.0", not "1"/"8"/"32"
    # -- coerce through float() on both sides rather than assume the bash
    # array's plain-integer strings match the JSON's key formatting.
    python3 -c "
import json
d = json.load(open('$1'))
key = str(float('$2'))
print(d['derived_rates_req_per_s'][key])
"
}

derived_service_time() {
    # derived_service_time <calibration_json>
    # The concurrency=1 probe's mean E2E time -- unqueued, so this is
    # service time directly (see scripts/calibrate.py's own docstring).
    # Used to scale the barge-in abort window per arm; see
    # BARGE_IN_MIN_FRAC/BARGE_IN_MAX_FRAC above.
    python3 -c "
import json
d = json.load(open('$1'))
print(d['measurement']['mean_service_time_s'])
"
}

run_repeat_cell() {
    # run_repeat_cell <bare_arm_label> <model> <rate> <conc>
    # bare_arm_label has no _pcoff/_pcon suffix -- repeats only ever cover
    # the prefix-off baseline, and scripts/analyze.py's repeat-detection
    # regex expects the bare arm name (repeat_fp16_c1_seed0.json, not
    # repeat_fp16_pcoff_c1_seed0.json). barge-in is always 0.0 here --
    # scripts/repeat_check.py hardcodes that -- so the scaled barge-in
    # window doesn't apply to repeat cells at all.
    local bare_arm="$1" model="$2" rate="$3" conc="$4"
    local out="$RESULTS_DIR/repeat_${bare_arm}_c${conc}.json"
    log "repeat (${REPEATS} seeds): arm=$bare_arm target_concurrency=$conc derived_rate=$rate"
    python3 scripts/repeat_check.py \
        --base-url "$BASE_URL" --model "$model" --label "${bare_arm}_c${conc}" \
        --mode open --rate "$rate" --duration "$LOAD_DURATION" --warmup 5 \
        --repeats "$REPEATS" --seed-start 0 \
        --out "$out"
}

run_open_loop_matrix() {
    # run_open_loop_matrix <arm_label> <bare_arm_label> <model> <calib_json> <use_repeats: true|false>
    #
    # arm_label is used for this run's own output filenames (carries the
    # _pcoff/_pcon suffix for fp16). bare_arm_label is only used when
    # use_repeats=true, for repeat_check.py's output filenames -- see
    # run_repeat_cell.
    local arm_label="$1" bare_arm_label="$2" model="$3" calib_json="$4" use_repeats="$5"
    local conc rate barge_in out service_time barge_in_min barge_in_max

    service_time=$(derived_service_time "$calib_json")
    barge_in_min=$(python3 -c "print($BARGE_IN_MIN_FRAC * $service_time)")
    barge_in_max=$(python3 -c "print($BARGE_IN_MAX_FRAC * $service_time)")
    log "scaled barge-in window: arm=$arm_label mean_service_time_s=$service_time -> min=${barge_in_min}s max=${barge_in_max}s"

    for conc in "${TARGET_CONCURRENCIES[@]}"; do
        rate=$(derived_rate "$calib_json" "$conc")
        for barge_in in "${BARGE_IN_FRACTIONS[@]}"; do
            local is_repeat_cell=false
            if [ "$use_repeats" = "true" ] && [ "$barge_in" = "0.0" ]; then
                local rc
                for rc in "${REPEAT_CONCURRENCIES[@]}"; do
                    [ "$conc" = "$rc" ] && is_repeat_cell=true
                done
            fi

            if [ "$is_repeat_cell" = "true" ]; then
                run_repeat_cell "$bare_arm_label" "$model" "$rate" "$conc"
                continue
            fi

            out="$RESULTS_DIR/${arm_label}_open_c${conc}_bargein${barge_in}.json"
            log "open-loop: arm=$arm_label target_concurrency=$conc derived_rate=$rate barge_in=$barge_in"
            python3 harness.py \
                --base-url "$BASE_URL" --model "$model" \
                --mode open --rate "$rate" --duration "$LOAD_DURATION" \
                --barge-in "$barge_in" \
                --barge-in-min "$barge_in_min" --barge-in-max "$barge_in_max" \
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
    # run_arm <bare_arm_label> <model> <extra vllm flags...>
    #
    # Prefix caching off is the baseline for every arm -- it's a large
    # lever on this multi-turn, eight-conversation workload, and leaving
    # it on for one arm but not another would make a cross-arm latency
    # gap unattributable to the thing actually being compared
    # (quantization). fp16 additionally runs with it on, as its own
    # swept dimension, not as a difference baked into which arm gets
    # which cache state. See docs/decisions.md.
    local bare_arm_label="$1" model="$2"; shift 2

    if [ "$bare_arm_label" = "fp16" ]; then
        local pc_state
        for pc_state in off on; do
            local flag="--no-enable-prefix-caching"
            [ "$pc_state" = "on" ] && flag="--enable-prefix-caching"
            local label="${bare_arm_label}_pc${pc_state}"
            local use_repeats="false"
            [ "$pc_state" = "off" ] && use_repeats="true"

            start_server "$label" "$model" "$@" "$flag"
            local calib_json
            calib_json=$(run_calibration "$model" "$label")
            run_open_loop_matrix "$label" "$bare_arm_label" "$model" "$calib_json" "$use_repeats"
            if [ "$pc_state" = "off" ]; then
                run_closed_loop_contrast "$bare_arm_label" "$model"
            fi
            stop_server
        done
    else
        start_server "$bare_arm_label" "$model" "$@" "--no-enable-prefix-caching"
        local calib_json
        calib_json=$(run_calibration "$model" "$bare_arm_label")
        run_open_loop_matrix "$bare_arm_label" "$bare_arm_label" "$model" "$calib_json" "true"
        run_closed_loop_contrast "$bare_arm_label" "$model"
        stop_server
    fi
}

main() {
    log "sweep starting: results -> $RESULTS_DIR"

    run_arm fp16 "Qwen/Qwen2.5-1.5B-Instruct"
    run_arm awq  "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    # --quantization fp8 is the flag verified for the vLLM version
    # actually installed (0.19.1, forced by driver incompatibility -- see
    # docs/decisions.md). It is NOT the same flag as the H100 run's
    # fp8_per_tensor, and not just a renamed version of it: on 0.19.1
    # this is confirmed W8A8 with dynamic activation scaling, not
    # weight-only static-scale (docs/decisions.md, "FP8 on vLLM 0.19.1").
    # If this ever runs against a different vLLM version, re-verify the
    # flag AND the resolved quantization semantics the same way before
    # trusting this line unchanged.
    run_arm fp8  "Qwen/Qwen2.5-1.5B-Instruct" --quantization fp8

    log "sweep complete"
}

main "$@"
