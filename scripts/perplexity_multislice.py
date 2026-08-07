"""
Perplexity across multiple, distinct wikitext-2 slices
(scripts/build_perplexity_slices.py) against an already-running vLLM
server -- one forced-decoding call per slice, same mechanism as
scripts/perplexity.py (max_tokens=0, echo=True, prompt_logprobs=0, no
sampling).

Why this script exists, not just repeats of one slice: forced-decoding on
a single fixed input is a deterministic computation (fixed weights, fixed
tokens, no sampling) -- repeating it measures whether the serving path is
bit-reproducible, not the uncertainty that actually applies to a
perplexity claim. It came back bit-identical every time
(results/h200/perplexity_fp16.json, perplexity_awq.json,
docs/decisions.md) -- a real methodology finding about this serving
path's determinism, but not an error bar; a band of zero can't tell you
whether an 8% gap between two arms is real. The uncertainty that matters
is how much perplexity moves across different text samples. That's what
scoring each of several distinct slices once, and taking mean/sd across
slices, actually measures.

Usage:
  python scripts/perplexity_multislice.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --label fp16 \
      --slices data/wikitext2_test_slices.json \
      --out results/h200/perplexity_multislice_fp16.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance


def run_once(base_url, model, token_ids, timeout_s):
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": 0,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.load(resp)
    latency_s = time.monotonic() - t0

    choice = body["choices"][0]
    prompt_logprobs = choice["prompt_logprobs"]
    if len(prompt_logprobs) != len(token_ids):
        raise RuntimeError(
            f"server returned {len(prompt_logprobs)} prompt_logprobs entries "
            f"for {len(token_ids)} input tokens -- expected one per token"
        )
    if prompt_logprobs[0] is not None:
        raise RuntimeError("expected prompt_logprobs[0] to be null (no preceding context)")

    logprobs, ranks = [], []
    for i, entry in enumerate(prompt_logprobs[1:], start=1):
        if entry is None:
            raise RuntimeError(f"unexpected null prompt_logprobs entry at position {i}")
        expected_id = str(token_ids[i])
        if expected_id not in entry:
            raise RuntimeError(
                f"prompt_logprobs at position {i} doesn't contain the actual "
                f"input token id {expected_id} -- got keys {list(entry.keys())}"
            )
        logprobs.append(entry[expected_id]["logprob"])
        ranks.append(entry[expected_id]["rank"])

    nll = -statistics.fmean(logprobs)
    return {
        "n_scored_tokens": len(logprobs),
        "nll": nll,
        "perplexity": math.exp(nll),
        "mean_rank": statistics.fmean(ranks),
        "latency_s": latency_s,
        "usage_prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="arm name, e.g. fp16 / awq")
    p.add_argument("--slices", default="data/wikitext2_test_slices.json")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default="results/perplexity_multislice.json")
    return p.parse_args()


def main():
    cfg = parse_args()

    with open(cfg.slices) as f:
        slices_meta = json.load(f)

    per_slice = []
    for s in slices_meta["slices"]:
        result = run_once(cfg.base_url, cfg.model, s["token_ids"], cfg.timeout)
        result["slice_index"] = s["index"]
        per_slice.append(result)
        print(
            f"  slice {s['index']}/{slices_meta['n_slices'] - 1}: "
            f"ppl={result['perplexity']:.4f} nll={result['nll']:.6f} "
            f"latency={result['latency_s']:.2f}s"
        )

    ppls = [r["perplexity"] for r in per_slice]
    nlls = [r["nll"] for r in per_slice]
    stats = {
        "perplexity": {
            "mean": statistics.fmean(ppls),
            "stdev": statistics.stdev(ppls) if len(ppls) > 1 else None,
            "min": min(ppls),
            "max": max(ppls),
        },
        "nll": {
            "mean": statistics.fmean(nlls),
            "stdev": statistics.stdev(nlls) if len(nlls) > 1 else None,
        },
        "n_slices": len(per_slice),
    }

    payload = {
        "provenance": provenance.capture(
            model=cfg.model,
            vllm_server_url=cfg.base_url,
            extra={
                "script": "perplexity_multislice.py",
                "label": cfg.label,
                "slices_source_repo": slices_meta["source_repo"],
                "slices_source_file": slices_meta["source_file"],
                "slices_source_file_sha256": slices_meta["source_file_sha256"],
                "slices_tokenizer": slices_meta["tokenizer"],
                "n_slices": slices_meta["n_slices"],
                "tokens_per_slice": slices_meta["tokens_per_slice"],
            },
        ),
        "config": vars(cfg),
        "stats": stats,
        "per_slice": per_slice,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"label": cfg.label, "stats": stats}, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
