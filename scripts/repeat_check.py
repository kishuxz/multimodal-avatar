"""
Run the same load config against an already-running server N times with
different seeds, and report mean/std of TTFT p50 and p99 across the
repeats -- the run-to-run noise band a single-run point estimate can't
tell you it needs.

A single harness.py run gives one TTFT p50 per arm/load point. Comparing
those single numbers across arms (e.g. "AWQ's 36.47ms vs fp16's 37.36ms")
answers a question this script's whole existence argues you can't answer
from n=1: is that gap real, or smaller than what one arm alone produces
run to run? This script measures the latter so the former can be judged
against it, rather than asserted.

Usage:
  python scripts/repeat_check.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --label fp16_c32 \
      --mode open --rate 498.6676 --duration 20 --warmup 5 \
      --repeats 5 --seed-start 0 \
      --out results/repeat_fp16_c32.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_once(base_url, model, mode, rate, concurrency, duration, warmup, seed, out):
    cmd = [
        sys.executable, "harness.py",
        "--base-url", base_url, "--model", model,
        "--mode", mode, "--duration", str(duration),
        "--barge-in", "0.0", "--warmup", str(warmup),
        "--seed", str(seed), "--out", out,
    ]
    if mode == "open":
        cmd += ["--rate", str(rate)]
    else:
        cmd += ["--concurrency", str(concurrency)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--mode", choices=["open", "closed"], default="open")
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default="results/repeat_check.json")
    return p.parse_args()


def main():
    cfg = parse_args()

    per_run_out_tmpl = cfg.out[:-len(".json")] if cfg.out.endswith(".json") else cfg.out
    ttft_p50s, ttft_p99s = [], []
    error_counts = []
    per_repeat = []

    for i in range(cfg.repeats):
        seed = cfg.seed_start + i
        run_out = f"{per_run_out_tmpl}_seed{seed}.json"
        result = run_once(
            cfg.base_url, cfg.model, cfg.mode, cfg.rate, cfg.concurrency,
            cfg.duration, cfg.warmup, seed, run_out,
        )
        s = result["summary"]
        ttft_p50s.append(s["ttft_s"]["p50"])
        ttft_p99s.append(s["ttft_s"]["p99"])
        error_counts.append(s["requests_error"])
        per_repeat.append({
            "seed": seed, "out": run_out,
            "ttft_p50_s": s["ttft_s"]["p50"], "ttft_p99_s": s["ttft_s"]["p99"],
            "requests_ok": s["requests_ok"], "requests_error": s["requests_error"],
        })
        print(f"  repeat {i+1}/{cfg.repeats} (seed={seed}): "
              f"ttft_p50={s['ttft_s']['p50']*1000:.2f}ms "
              f"ttft_p99={s['ttft_s']['p99']*1000:.2f}ms "
              f"errors={s['requests_error']}")

    stats = {
        "ttft_p50_s": {
            "mean": statistics.fmean(ttft_p50s),
            "stdev": statistics.stdev(ttft_p50s) if len(ttft_p50s) > 1 else None,
            "min": min(ttft_p50s), "max": max(ttft_p50s),
        },
        "ttft_p99_s": {
            "mean": statistics.fmean(ttft_p99s),
            "stdev": statistics.stdev(ttft_p99s) if len(ttft_p99s) > 1 else None,
            "min": min(ttft_p99s), "max": max(ttft_p99s),
        },
        "total_errors": sum(error_counts),
    }

    payload = {
        "provenance": provenance.capture(
            model=cfg.model, vllm_server_url=cfg.base_url,
            extra={"script": "repeat_check.py", "label": cfg.label},
        ),
        "config": vars(cfg),
        "stats": stats,
        "per_repeat": per_repeat,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"label": cfg.label, "stats": stats}, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
