"""
Derive an open-loop arrival rate for each target concurrency via Little's
Law: L = lambda * W, so lambda = L / W.

W (mean service time) is measured with a concurrency=1 closed-loop probe
against harness.py. At concurrency 1 there is no queueing, so E2E time
*is* service time, not service time plus wait -- the "low load"
measurement this project's methodology calls for. Calibration has to run
before any load test because service time differs by arm: a fixed
arrival rate does not mean a fixed target concurrency once the arms
don't all take the same time to answer.

This produces a target OFFERED concurrency, not a guaranteed realized
one -- at high load, queueing delay under an open-loop Poisson process
adds to time-in-system, so the actual average number of requests in
flight can run higher than L as a server approaches saturation. See
docs/decisions.md for why this repo holds offered concurrency (not
arrival rate) constant across arms when comparing them, and what that
does and doesn't claim.

Usage:
  python scripts/calibrate.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --label fp16 \
      --duration 30 --target-concurrency 1 8 32 \
      --out results/calibration_fp16.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_calibration_probe(base_url, model, duration, warmup, seed, probe_out):
    """Runs harness.py at concurrency=1, closed loop, no barge-in -- the
    lowest-load shape this harness can produce, so E2E ~= service time."""
    cmd = [
        sys.executable, "harness.py",
        "--base-url", base_url,
        "--model", model,
        "--mode", "closed",
        "--concurrency", "1",
        "--duration", str(duration),
        "--barge-in", "0.0",
        "--warmup", str(warmup),
        "--seed", str(seed),
        "--out", probe_out,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with open(os.path.join(REPO_ROOT, probe_out) if not os.path.isabs(probe_out) else probe_out) as f:
        return json.load(f)


def derive_rates(mean_service_time_s, target_concurrencies):
    return {str(l): round(l / mean_service_time_s, 4) for l in target_concurrencies}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="arm name, e.g. fp16 / awq / fp8")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-concurrency", type=float, nargs="+", default=[1.0, 8.0, 32.0])
    p.add_argument("--out", default="results/calibration.json")
    return p.parse_args()


def main():
    cfg = parse_args()

    probe_out = cfg.out[:-len(".json")] + "_probe.json" if cfg.out.endswith(".json") else cfg.out + ".probe.json"

    probe = run_calibration_probe(
        cfg.base_url, cfg.model, cfg.duration, cfg.warmup, cfg.seed, probe_out,
    )

    e2e = probe["summary"]["e2e_s"]
    mean_service_time_s = e2e["mean"]
    n = e2e["n"]

    if mean_service_time_s is None or n < 3:
        raise RuntimeError(
            f"calibration probe produced too few samples to trust (n={n}); "
            f"increase --duration or check the server is actually reachable"
        )

    derived_rates = derive_rates(mean_service_time_s, cfg.target_concurrency)

    payload = {
        "provenance": provenance.capture(
            model=cfg.model, vllm_server_url=cfg.base_url,
            extra={"script": "calibrate.py", "arm_label": cfg.label},
        ),
        "config": vars(cfg),
        "measurement": {
            "method": "closed-loop, concurrency=1, barge-in=0.0 -- no "
                      "queueing at that concurrency, so E2E is service "
                      "time directly, not service time plus wait",
            "probe_result_ref": probe_out,
            "mean_service_time_s": mean_service_time_s,
            "service_time_p50_s": e2e["p50"],
            "service_time_p95_s": e2e["p95"],
            "n_samples": n,
        },
        "derived_rates_req_per_s": derived_rates,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "arm": cfg.label,
        "mean_service_time_s": mean_service_time_s,
        "n_samples": n,
        "derived_rates_req_per_s": derived_rates,
    }, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
