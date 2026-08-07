"""
Phase 8: realized concurrency during an already-committed load-test run
-- how many requests were actually in flight at once, sampled regularly
across the run, vs. the ~32 concurrency target that arm's arrival rate
was calibrated for. Reads only committed results/*.json; no server, no
GPU.

Scaled arrival rate targets a concurrency via Little's Law from each
arm's own *unqueued* (concurrency=1) service time (scripts/calibrate.py)
-- under real queueing near saturation, the realized average can run
higher than that target, and by how much differs by arm if the arms
aren't equally close to their own saturation point at the calibrated
rate. This script measures that directly from request-level
arrival/e2e timestamps already in each run file, rather than assuming
"concurrency ~= 32" was equally true for every arm.

Usage:
  python scripts/realized_concurrency.py \
      --files results/h200/repeat_fp16_c32_seed{0,1,2,3,4}.json \
      --label fp16
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import statistics


def realized_concurrency(path, n_samples=500):
    with open(path) as f:
        d = json.load(f)
    warmup = d["config"]["warmup"]
    kept = [r for r in d["requests"][warmup:] if r["error"] is None]
    intervals = [(r["arrival"], r["arrival"] + r["e2e"]) for r in kept if r.get("e2e") is not None]
    if not intervals:
        return None
    starts = sorted(s for s, e in intervals)
    ends = sorted(e for s, e in intervals)
    t_min, t_max = starts[0], max(ends)
    step = (t_max - t_min) / n_samples
    samples = []
    t = t_min
    while t < t_max:
        in_flight = bisect.bisect_right(starts, t) - bisect.bisect_right(ends, t)
        samples.append(in_flight)
        t += step
    return {"mean": statistics.fmean(samples), "max": max(samples), "n_samples": len(samples)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True, help="one or more results/*.json paths (globs expanded)")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    paths = []
    for pattern in cfg.files:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    per_file = []
    for path in paths:
        stats = realized_concurrency(path)
        if stats is None:
            continue
        stats["file"] = path
        per_file.append(stats)
        print(f"  {path}: mean={stats['mean']:.2f} max={stats['max']}")

    means = [f["mean"] for f in per_file]
    maxes = [f["max"] for f in per_file]
    summary = {
        "label": cfg.label,
        "n_files": len(per_file),
        "mean_realized_concurrency": statistics.fmean(means) if means else None,
        "mean_realized_concurrency_stdev": statistics.stdev(means) if len(means) > 1 else None,
        "max_realized_concurrency_across_files": max(maxes) if maxes else None,
        "per_file": per_file,
    }

    import os
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
