"""
Phase 8: kernel-level time breakdown for an arm under real, sustained
c~=32 open-loop load -- the actual new measurement behind the AWQ
bandwidth-hypothesis profiling pass. Not a synthetic offline batch:
`harness.py` drives the server exactly the way the latency sweep did,
at that arm's own calibrated c~=32 rate, and the profiling window
(torch.profiler, via vLLM's own --profiler-config + /start_profile /
/stop_profile) sits inside a longer steady-state run so the captured
kernels reflect real traffic, not cold-start effects.

`ncu`/`nsys` (direct hardware-counter roofline tools) are blocked on
this pod at the hypervisor level -- confirmed, not fixable from inside
the container even as root; see docs/decisions.md, "Phase 8: profiling
AWQ's c~=32 advantage." torch.profiler uses CUPTI's activity/tracing
API, a different, unrestricted path -- it gives real kernel names,
counts, and durations, just not achieved-bandwidth or SM-utilization
percentages the way `ncu` counters would.

Usage (server must already be running with
--profiler-config.profiler torch --profiler-config.torch_profiler_dir
<dir>, matching --profiler-dir below):
  python scripts/profile_awq_bandwidth.py \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct \
      --label fp16 --rate 475.0453 --profiler-dir /workspace/profiles/fp16 \
      --out results/h200/profile_awq_bandwidth_fp16.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Substring match against the kernel name, checked in order -- first
# match wins. Built from inspecting a live trace before writing this
# (docs/decisions.md), not guessed from kernel-name conventions.
KERNEL_CATEGORIES = [
    ("gemm", ["nvjet_", "cutlass::gemm", "cublaslt", "cublas", "marlin", "machete",
              "cutlass_tensorop", "gemm_kernel"]),
    ("attention", ["flashattn", "flash_attn", "flash::", "softmax", "attention"]),
    ("kv_cache", ["reshape_and_cache"]),
    ("norm_activation", ["rsqrt", "silu", "triton_red_fused", "triton_poi_fused",
                          "layernorm", "rmsnorm"]),
    ("sampling_sort", ["radixsort", "topk", "sort"]),
    ("memcpy_memset", []),  # matched by trace `cat`, not name -- see below
]


def categorize(name, cat):
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy_memset"
    lname = name.lower()
    for label, substrings in KERNEL_CATEGORIES:
        for s in substrings:
            if s in lname:
                return label
    return "other"


def call_profile_endpoint(server_root, path):
    # /stop_profile blocks until vLLM finishes exporting the trace, which
    # is slow under real load -- a 1.5s window at c~=32 produced a
    # 70-90MB gzipped trace and took well over 30s to flush server-side
    # (confirmed: the trace file existed and was complete even after this
    # client gave up waiting the first time this ran with a 30s timeout).
    # 300s is a generous ceiling, not a tuned value.
    req = urllib.request.Request(f"{server_root.rstrip('/')}/{path}", method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status


def run_harness(base_url, model, rate, duration, warmup, seed, out):
    cmd = [
        sys.executable, "harness.py",
        "--base-url", base_url, "--model", model,
        "--mode", "open", "--rate", str(rate), "--duration", str(duration),
        "--barge-in", "0.0", "--warmup", str(warmup), "--seed", str(seed),
        "--out", out,
    ]
    return subprocess.Popen(cmd, cwd=REPO_ROOT)


def find_latest_rank_trace(profiler_dir):
    candidates = [
        os.path.join(profiler_dir, f) for f in os.listdir(profiler_dir)
        if f.startswith("rank0.") and f.endswith(".pt.trace.json.gz")
    ]
    if not candidates:
        raise RuntimeError(f"no rank0 trace file found in {profiler_dir}")
    return max(candidates, key=os.path.getmtime)


def parse_trace(trace_path):
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]

    per_kernel = {}
    for k in kernels:
        name = k["name"]
        dur = k.get("dur", 0)
        entry = per_kernel.setdefault(name, {"total_us": 0.0, "count": 0, "category": categorize(name, k.get("cat"))})
        entry["total_us"] += dur
        entry["count"] += 1

    total_us = sum(e["total_us"] for e in per_kernel.values())
    per_category = {}
    for name, e in per_kernel.items():
        cat = e["category"]
        per_category[cat] = per_category.get(cat, 0.0) + e["total_us"]

    top_kernels = sorted(
        ({"name": n, **e} for n, e in per_kernel.items()),
        key=lambda x: -x["total_us"],
    )[:20]
    for k in top_kernels:
        k["pct_of_total"] = k["total_us"] / total_us * 100 if total_us else None

    category_summary = {
        cat: {"total_us": us, "pct_of_total": us / total_us * 100 if total_us else None}
        for cat, us in sorted(per_category.items(), key=lambda x: -x[1])
    }

    return {
        "trace_file": os.path.basename(trace_path),
        "total_kernel_time_us": total_us,
        "n_distinct_kernels": len(per_kernel),
        "n_kernel_launches": len(kernels),
        "category_summary": category_summary,
        "top_kernels": top_kernels,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--rate", type=float, required=True, help="calibrated c~=32 open-loop rate for this arm")
    p.add_argument("--profiler-dir", required=True, help="must match the server's --profiler-config.torch_profiler_dir")
    p.add_argument("--total-duration", type=float, default=14.0, help="harness.py run length -- warmup + profile window + tail")
    p.add_argument("--warmup-before-profile", type=float, default=6.0, help="seconds of load before starting the profiler")
    p.add_argument("--profile-window", type=float, default=1.5, help="seconds the profiler stays on")
    p.add_argument("--harness-warmup-requests", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    server_root = cfg.base_url.rsplit("/v1", 1)[0]
    harness_out = cfg.out.replace(".json", "_harness.json")

    proc = run_harness(cfg.base_url, cfg.model, cfg.rate, cfg.total_duration,
                        cfg.harness_warmup_requests, cfg.seed, harness_out)

    time.sleep(cfg.warmup_before_profile)
    call_profile_endpoint(server_root, "start_profile")
    print(f"  profiler on, window={cfg.profile_window}s")
    time.sleep(cfg.profile_window)
    call_profile_endpoint(server_root, "stop_profile")
    print("  profiler off, waiting for trace to flush and harness.py to finish")
    time.sleep(3.0)  # trace export is async; give it a moment before reading

    proc.wait(timeout=60)

    with open(harness_out) as f:
        harness_result = json.load(f)

    trace_path = find_latest_rank_trace(cfg.profiler_dir)
    trace_summary = parse_trace(trace_path)

    payload = {
        "provenance": provenance.capture(
            model=cfg.model,
            vllm_server_url=cfg.base_url,
            extra={
                "script": "profile_awq_bandwidth.py",
                "label": cfg.label,
                "rate": cfg.rate,
                "profiler_dir": cfg.profiler_dir,
                "warmup_before_profile_s": cfg.warmup_before_profile,
                "profile_window_s": cfg.profile_window,
            },
        ),
        "config": vars(cfg),
        "harness_summary": harness_result.get("summary"),
        "trace_summary": trace_summary,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"label": cfg.label, "category_summary": trace_summary["category_summary"]}, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
