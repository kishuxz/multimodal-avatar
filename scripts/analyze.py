"""
Phase 5: reads results/*.json committed by the Phase 1-3 sweep (and its
calibration/repeat passes) and emits the arm x load x barge-in x
prefix-caching matrix as markdown, plus six plots. Nothing here talks to
a server -- every number traces back to a JSON file already in this repo,
which is the point of this phase: it runs with no GPU and no pod.

Where a config was repeat-validated (scripts/repeat_check.py, 5 seeds,
concurrency ~= 1 and ~= 32, prefix caching off, barge-in 0.0), TTFT p50/p99
mean+/-stdev are read directly from that script's own committed output
(results/repeat_<arm>_c<N>.json) rather than recomputed here, so this
report never disagrees with docs/decisions.md's already-checked numbers.
Every other repeat-covered column (p95, ITL, tokens/sec, errors) is
aggregated from the 5 underlying per-seed run files the same way.
Everywhere else, a cell is a single run and reported as one.

Usage:
  python scripts/analyze.py
  (reads results/*.json, writes results/summary.md and plots/*.png)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Filenames are sweep.sh's own naming convention -- see scripts/sweep.sh.
# fp16's closed-loop contrast run is named plain "fp16_closed_c8.json", not
# "fp16_pcoff_closed_c8.json" -- the pcon/pcoff token only ever appears on
# fp16's open-loop files. Matched here as its own alternative rather than
# folded into fp16_pcoff, so a future bare "fp16_open_*" (if one ever
# shows up) doesn't silently get treated as a known-off open-loop cell.
RUN_RE = re.compile(
    r"^(?P<arm>fp16_pcon|fp16_pcoff|fp16|awq|fp8)_(?P<mode>open|closed)_c(?P<conc>\d+)"
    r"(?:_bargein(?P<bargein>[\d.]+))?\.json$"
)
REPEAT_SEED_RE = re.compile(
    r"^repeat_(?P<arm>fp16|awq|fp8)_c(?P<conc>\d+)_seed(?P<seed>\d+)\.json$"
)
REPEAT_SUMMARY_RE = re.compile(r"^repeat_(?P<arm>fp16|awq|fp8)_c(?P<conc>\d+)\.json$")

# filename arm token -> (canonical arm, prefix caching on/off). AWQ and FP8
# only ever run with prefix caching off (docs/decisions.md: "prefix caching
# off is the cross-arm baseline"), so they have no separate pcon/pcoff form.
ARM_TOKEN = {
    "fp16_pcoff": ("fp16", False),
    "fp16_pcon": ("fp16", True),
    "fp16": ("fp16", False),
    "awq": ("awq", False),
    "fp8": ("fp8", False),
}

ARM_ORDER = [("fp16", False), ("fp16", True), ("awq", False), ("fp8", False)]
CONC_ORDER = [1, 8, 32]
BARGEIN_ORDER = [0.0, 0.25]


def pct(xs, p):
    # Mirrors harness.py's own pct() (linear-interpolated nearest-rank) --
    # duplicated rather than imported, since harness.py requires aiohttp
    # (pod-only; see requirements-pod.txt) purely to reach this six-line
    # function, which would drag a live-server dependency into a script
    # that is supposed to run with no server and no GPU.
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ms(x):
    return None if x is None else x * 1000.0


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_runs(results_dir):
    """One entry per single harness.py run file (open or closed loop)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        m = RUN_RE.match(name)
        if not m:
            continue
        arm, prefix_on = ARM_TOKEN[m.group("arm")]
        bargein = float(m.group("bargein")) if m.group("bargein") else 0.0
        runs.append({
            "path": path, "file": name,
            "arm": arm, "prefix_caching": prefix_on,
            "mode": m.group("mode"), "concurrency_target": int(m.group("conc")),
            "barge_in": bargein,
        })
    return runs


def discover_repeat_seeds(results_dir):
    """(arm, concurrency_target) -> list of the 5 per-seed run file paths.
    Repeats only ever cover prefix-off, open-loop, barge-in 0.0."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "repeat_*_seed*.json"))):
        m = REPEAT_SEED_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (m.group("arm"), int(m.group("conc")))
        groups.setdefault(key, []).append(path)
    return groups


def discover_repeat_summaries(results_dir):
    """(arm, concurrency_target) -> repeat_check.py's own stats block,
    already mean/stdev'd across 5 seeds -- the numbers docs/decisions.md
    cites directly. Kept separate from the seed files above so TTFT
    p50/p99 always come from repeat_check.py's own arithmetic, not a
    second recompute of it."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "repeat_*.json"))):
        name = os.path.basename(path)
        if REPEAT_SEED_RE.match(name):
            continue
        m = REPEAT_SUMMARY_RE.match(name)
        if not m:
            continue
        key = (m.group("arm"), int(m.group("conc")))
        out[key] = load_json(path)
    return out


# --------------------------------------------------------------------------
# Cell construction: one row per (arm, prefix_caching, mode, concurrency, barge_in)
# --------------------------------------------------------------------------

def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], None
    return statistics.fmean(xs), statistics.stdev(xs)


def single_run_cell(run):
    d = load_json(run["path"])
    s = d["summary"]
    return {
        **run,
        "runs": 1,
        "rate": d["config"].get("rate") if run["mode"] == "open" else None,
        "n_requests": s["requests_ok"],
        "ttft_p50_ms": (ms(s["ttft_s"]["p50"]), None),
        "ttft_p95_ms": (ms(s["ttft_s"]["p95"]), None),
        "ttft_p99_ms": (ms(s["ttft_s"]["p99"]), None),
        "itl_p50_ms": (ms(s["itl_s"]["p50"]), None),
        "itl_p95_ms": (ms(s["itl_s"]["p95"]), None),
        "tokens_per_sec": (s["output_tokens_per_sec"], None),
        "errors": (s["requests_error"], None),
    }


def repeat_cell(run, seed_paths, repeat_stats):
    seed_summaries = [load_json(p)["summary"] for p in seed_paths]
    rate = load_json(run["path"])["config"].get("rate")
    n_requests = sum(s["requests_ok"] for s in seed_summaries)
    st = repeat_stats["stats"]
    return {
        **run,
        "runs": len(seed_paths),
        "rate": rate,
        "n_requests": n_requests,
        "ttft_p50_ms": (ms(st["ttft_p50_s"]["mean"]), ms(st["ttft_p50_s"]["stdev"])),
        "ttft_p99_ms": (ms(st["ttft_p99_s"]["mean"]), ms(st["ttft_p99_s"]["stdev"])),
        "ttft_p95_ms": mean_sd([ms(s["ttft_s"]["p95"]) for s in seed_summaries]),
        "itl_p50_ms": mean_sd([ms(s["itl_s"]["p50"]) for s in seed_summaries]),
        "itl_p95_ms": mean_sd([ms(s["itl_s"]["p95"]) for s in seed_summaries]),
        "tokens_per_sec": mean_sd([s["output_tokens_per_sec"] for s in seed_summaries]),
        "errors": mean_sd([s["requests_error"] for s in seed_summaries]),
    }


def build_cells(results_dir):
    runs = discover_runs(results_dir)
    seed_groups = discover_repeat_seeds(results_dir)
    repeat_stats = discover_repeat_summaries(results_dir)

    cells = {}
    for run in runs:
        repeat_key = (run["arm"], run["concurrency_target"])
        is_repeat_covered = (
            run["mode"] == "open" and not run["prefix_caching"]
            and run["barge_in"] == 0.0 and repeat_key in seed_groups
        )
        if is_repeat_covered:
            cell = repeat_cell(run, seed_groups[repeat_key], repeat_stats[repeat_key])
        else:
            cell = single_run_cell(run)
        key = (run["arm"], run["prefix_caching"], run["mode"],
               run["concurrency_target"], run["barge_in"])
        cells[key] = cell
    return cells


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def fmt(pair, decimals=2):
    val, sd = pair
    if val is None:
        return "--"
    if sd is None:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} ± {sd:.{decimals}f}"


def sorted_cells(cells):
    def sort_key(item):
        (arm, prefix, mode, conc, bargein), _ = item
        arm_idx = ARM_ORDER.index((arm, prefix))
        mode_idx = 0 if mode == "open" else 1
        return (arm_idx, mode_idx, conc, bargein)
    return sorted(cells.items(), key=sort_key)


def main_table(cells):
    lines = [
        "| Arm | Prefix caching | Mode | Concurrency (target) | Barge-in | "
        "Rate (req/s) | Runs | n | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | "
        "ITL p50 (ms) | ITL p95 (ms) | Tokens/s | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (arm, prefix, mode, conc, bargein), c in sorted_cells(cells):
        rate = f"{c['rate']:.2f}" if c["rate"] is not None else "n/a (closed-loop)"
        lines.append(
            f"| {arm} | {'on' if prefix else 'off'} | {mode} | {conc} | {bargein} | "
            f"{rate} | {c['runs']} | {c['n_requests']} | "
            f"{fmt(c['ttft_p50_ms'])} | {fmt(c['ttft_p95_ms'])} | {fmt(c['ttft_p99_ms'])} | "
            f"{fmt(c['itl_p50_ms'])} | {fmt(c['itl_p95_ms'])} | "
            f"{fmt(c['tokens_per_sec'], 1)} | {fmt(c['errors'], 1)} |"
        )
    return "\n".join(lines)


def prefix_caching_table(cells):
    lines = [
        "| Concurrency (target) | Metric | Prefix caching off | Prefix caching on | Change |",
        "|---|---|---|---|---|",
    ]
    for conc in CONC_ORDER:
        off = cells.get(("fp16", False, "open", conc, 0.0))
        on = cells.get(("fp16", True, "open", conc, 0.0))
        if not off or not on:
            continue
        for label, key in [("TTFT p50 (ms)", "ttft_p50_ms"), ("TTFT p99 (ms)", "ttft_p99_ms")]:
            off_v, on_v = off[key][0], on[key][0]
            change = f"{(on_v - off_v) / off_v * 100:+.1f}%" if off_v else "--"
            lines.append(f"| {conc} | {label} | {fmt(off[key])} | {fmt(on[key])} | {change} |")
    return "\n".join(lines)


def open_vs_closed_table(cells):
    lines = [
        "| Arm | Metric | Open-loop (honest) | Closed-loop (self-throttling) | "
        "How much better closed-loop falsely looks |",
        "|---|---|---|---|---|",
    ]
    for arm in ["fp16", "awq", "fp8"]:
        open_cell = cells.get((arm, False, "open", 8, 0.0))
        closed_cell = cells.get((arm, False, "closed", 8, 0.0))
        if not open_cell or not closed_cell:
            continue
        for label, key in [("TTFT p50 (ms)", "ttft_p50_ms"), ("TTFT p99 (ms)", "ttft_p99_ms")]:
            o_v, c_v = open_cell[key][0], closed_cell[key][0]
            pct_lower = (o_v - c_v) / o_v * 100 if o_v else None
            delta = f"closed reads {pct_lower:.1f}% lower" if pct_lower is not None else "--"
            lines.append(f"| {arm} | {label} | {fmt(open_cell[key])} | {fmt(closed_cell[key])} | {delta} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def load_kept_requests(path):
    """requests[] includes the warmup prefix harness.py itself discards
    (see summarize() in harness.py) -- slice it out here the same way,
    and drop error rows, so every plot reads the same population the
    committed summary stats do."""
    d = load_json(path)
    warmup = d["config"]["warmup"]
    kept = d["requests"][warmup:]
    return d, [r for r in kept if r["error"] is None]


def plot_ttft_vs_arrival_rate(cells, out_path):
    # Only the highest-rate point is labeled: at the two low-rate points
    # all four series sit within ~2ms of each other, so per-point labels
    # there collide into an unreadable smear rather than adding
    # information -- the table has the exact numbers. The high-rate point
    # is where the series actually separate, which is the point this plot
    # exists to make.
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, prefix in ARM_ORDER:
        xs, ys = [], []
        for conc in CONC_ORDER:
            c = cells.get((arm, prefix, "open", conc, 0.0))
            if c and c["rate"] is not None and c["ttft_p50_ms"][0] is not None:
                xs.append(c["rate"])
                ys.append(c["ttft_p50_ms"][0])
        if xs:
            label = f"{arm} (prefix {'on' if prefix else 'off'})" if arm == "fp16" else arm
            ax.plot(xs, ys, marker="o", label=label)
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                         xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Offered arrival rate (req/s, log scale)")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("TTFT p50 vs arrival rate, open-loop, barge-in off\n"
                  "(closed-loop cannot show this curve -- it self-throttles instead of queueing)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prefix_caching_effect(cells, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (label, key) in zip(axes, [("TTFT p50 (ms)", "ttft_p50_ms"), ("TTFT p99 (ms)", "ttft_p99_ms")]):
        off_vals, on_vals = [], []
        for conc in CONC_ORDER:
            off = cells.get(("fp16", False, "open", conc, 0.0))
            on = cells.get(("fp16", True, "open", conc, 0.0))
            off_vals.append(off[key][0] if off else 0)
            on_vals.append(on[key][0] if on else 0)
        x = range(len(CONC_ORDER))
        width = 0.35
        ax.bar([i - width / 2 for i in x], off_vals, width, label="prefix caching off")
        ax.bar([i + width / 2 for i in x], on_vals, width, label="prefix caching on")
        for i, (o, n) in enumerate(zip(off_vals, on_vals)):
            if o:
                ax.annotate(f"{(n - o) / o * 100:+.0f}%", (i, max(o, n)),
                             ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"c≈{c}" for c in CONC_ORDER])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Prefix caching on vs off, fp16 -- the largest lever in this sweep\n"
                  "(single runs, not yet repeat-validated -- see docs/decisions.md)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttft_distribution_per_arm(results_dir, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, conc in zip(axes, CONC_ORDER):
        for arm, prefix in [("fp16", False), ("awq", False), ("fp8", False)]:
            seed_paths = sorted(glob.glob(
                os.path.join(results_dir, f"repeat_{arm}_c{conc}_seed*.json")))
            if seed_paths:
                paths = seed_paths
            else:
                fname = f"{'fp16_pcoff' if arm == 'fp16' else arm}_open_c{conc}_bargein0.0.json"
                paths = [os.path.join(results_dir, fname)]
            vals = []
            for p in paths:
                if os.path.exists(p):
                    _, kept = load_kept_requests(p)
                    vals.extend(r["ttft"] * 1000 for r in kept if r["ttft"] is not None)
            if vals:
                ax.hist(vals, bins=40, alpha=0.5, density=True, label=f"{arm} (n={len(vals)})")
        ax.set_title(f"concurrency ≈ {conc}")
        ax.set_xlabel("TTFT (ms)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    fig.suptitle("TTFT distribution per arm, prefix caching off, barge-in off\n"
                  "(c1/c32 pool all 5 repeat seeds; c8 has no repeats, single run)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def abort_window_itls(path, window_s=0.2):
    """Splits ITL samples (ms) from non-aborted requests into two groups by
    whether their arrival fell in a time window that also contained an
    abort event elsewhere. Excludes the aborting requests' own ITLs --
    those are truncated by definition and don't speak to collateral
    impact on the other requests sharing that window. This is the
    "effect on OTHER in-flight requests" harness.py's own docstring
    names as the reason barge-in records what it does."""
    _, kept = load_kept_requests(path)
    if not kept:
        return [], [], 0
    t0 = min(r["arrival"] for r in kept)

    def window_of(r):
        return int((r["arrival"] - t0) // window_s)

    abort_windows = {window_of(r) for r in kept if r["aborted"]}
    with_abort, without_abort = [], []
    for r in kept:
        if r["aborted"]:
            continue
        bucket = with_abort if window_of(r) in abort_windows else without_abort
        bucket.extend(x * 1000 for x in r["itls"])
    return with_abort, without_abort, len(abort_windows)


def plot_itl_abort_windows(results_dir, out_path):
    # Barge-in only overlaps request lifetimes at concurrency ~= 32 in this
    # sweep -- at c1/c8 requests finish faster than the sampled abort delay
    # (0.3-1.2s, harness.py's --barge-in-min/--barge-in-max), so no abort
    # ever fires there. Restricting to c32 is a finding, not a simplification.
    arms = [("fp16", "fp16_pcoff_open_c32_bargein0.25.json"),
            ("awq", "awq_open_c32_bargein0.25.json"),
            ("fp8", "fp8_open_c32_bargein0.25.json")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (arm, fname) in zip(axes, arms):
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        with_abort, without_abort, n_windows = abort_window_itls(path)
        data = [without_abort, with_abort]
        ax.boxplot(data, tick_labels=[f"no abort\n(n={len(without_abort)})",
                                       f"abort in window\n(n={len(with_abort)})"],
                    showfliers=False)
        ax.set_title(f"{arm} (c≈32, {n_windows} abort windows)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ITL (ms), non-aborted requests only")
    fig.suptitle("ITL of OTHER in-flight requests: windows with a barge-in abort vs without\n"
                  "(200ms windows, concurrency ≈ 32, barge-in 0.25 -- the only combination "
                  "where aborts overlap request lifetimes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_comparison(cells, out_path):
    # The other plots each make one comparison in isolation (fp16 on vs
    # off; the arrival-rate curve). Neither puts prefix caching's effect
    # size in the same frame as quantization's, so "prefix caching is the
    # largest lever" -- the headline claim -- isn't visually provable from
    # either alone; a reader has to cross-reference the table. This plot
    # exists only to make that one comparison legible from a single image.
    baseline = cells.get(("fp16", False, "open", 32, 0.0))
    levers = [
        ("prefix caching\n(fp16 on vs off)", cells.get(("fp16", True, "open", 32, 0.0))),
        ("AWQ\n(vs fp16)", cells.get(("awq", False, "open", 32, 0.0))),
        ("FP8\n(vs fp16)", cells.get(("fp8", False, "open", 32, 0.0))),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [l for l, c in levers if c]
    p50_deltas = [(c["ttft_p50_ms"][0] - baseline["ttft_p50_ms"][0]) / baseline["ttft_p50_ms"][0] * 100
                  for l, c in levers if c]
    p99_deltas = [(c["ttft_p99_ms"][0] - baseline["ttft_p99_ms"][0]) / baseline["ttft_p99_ms"][0] * 100
                  for l, c in levers if c]
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], p50_deltas, width, label="TTFT p50")
    ax.bar([i + width / 2 for i in x], p99_deltas, width, label="TTFT p99")
    for i, (p50, p99) in enumerate(zip(p50_deltas, p99_deltas)):
        ax.annotate(f"{p50:+.0f}%", (i - width / 2, p50), ha="center",
                     va="bottom" if p50 >= 0 else "top", fontsize=8)
        ax.annotate(f"{p99:+.0f}%", (i + width / 2, p99), ha="center",
                     va="bottom" if p99 >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change vs fp16, prefix caching off (%)")
    ax.set_title("Effect size at concurrency ≈ 32: prefix caching vs quantization\n"
                  "(prefix caching single-run; AWQ/FP8 repeat-validated)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_open_vs_closed(cells, out_path):
    arms = ["fp16", "awq", "fp8"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (label, key) in zip(axes, [("TTFT p50 (ms)", "ttft_p50_ms"), ("TTFT p99 (ms)", "ttft_p99_ms")]):
        open_vals, closed_vals = [], []
        for arm in arms:
            o = cells.get((arm, False, "open", 8, 0.0))
            c = cells.get((arm, False, "closed", 8, 0.0))
            open_vals.append(o[key][0] if o else 0)
            closed_vals.append(c[key][0] if c else 0)
        x = range(len(arms))
        width = 0.35
        ax.bar([i - width / 2 for i in x], open_vals, width, label="open-loop (honest)")
        ax.bar([i + width / 2 for i in x], closed_vals, width, label="closed-loop (self-throttling)")
        for i, (o, c) in enumerate(zip(open_vals, closed_vals)):
            if o:
                ax.annotate(f"{(o - c) / o * 100:.0f}% lower", (i, max(o, c)),
                             ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(arms)
        ax.set_ylim(top=max(open_vals + closed_vals) * 1.2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Open-loop vs closed-loop at the same nominal concurrency (c≈8)\n"
                  "closed-loop hides queueing by construction -- this is how much")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--plots-dir", default=os.path.join(REPO_ROOT, "plots"))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "summary.md"))
    return p.parse_args()


def main():
    cfg = parse_args()
    os.makedirs(cfg.plots_dir, exist_ok=True)

    cells = build_cells(cfg.results_dir)

    plot_ttft_vs_arrival_rate(cells, os.path.join(cfg.plots_dir, "ttft_vs_arrival_rate.png"))
    plot_prefix_caching_effect(cells, os.path.join(cfg.plots_dir, "prefix_caching_effect.png"))
    plot_effect_size_comparison(cells, os.path.join(cfg.plots_dir, "effect_size_comparison.png"))
    plot_ttft_distribution_per_arm(cfg.results_dir, os.path.join(cfg.plots_dir, "ttft_distribution_per_arm.png"))
    plot_itl_abort_windows(cfg.results_dir, os.path.join(cfg.plots_dir, "itl_abort_windows.png"))
    plot_open_vs_closed(cells, os.path.join(cfg.plots_dir, "open_vs_closed_loop.png"))

    md = f"""# Results summary

Generated by `scripts/analyze.py` from the JSON files committed under
`results/`. Regenerate with `make analyze` (or `python3 scripts/analyze.py`)
-- never hand-edit this file. "Runs" > 1 means the cell is repeat-validated
(`scripts/repeat_check.py`, 5 seeds); its columns are mean +/- stdev across
those runs, not a single-run point estimate. Everything else is n=1 and
should be read with that caveat.

## Full matrix: arm x load x barge-in x prefix caching

{main_table(cells)}

## Prefix caching effect (fp16, the largest lever in this sweep)

{prefix_caching_table(cells)}

## Open-loop vs closed-loop at the same nominal concurrency (c≈8)

{open_vs_closed_table(cells)}

## Plots

- `plots/ttft_vs_arrival_rate.png`
- `plots/prefix_caching_effect.png`
- `plots/effect_size_comparison.png`
- `plots/ttft_distribution_per_arm.png`
- `plots/itl_abort_windows.png`
- `plots/open_vs_closed_loop.png`
"""
    with open(cfg.out, "w") as f:
        f.write(md)
    print(f"wrote {cfg.out}")
    print(f"wrote 6 plots to {cfg.plots_dir}/")


if __name__ == "__main__":
    main()
