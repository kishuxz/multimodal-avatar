"""
Perplexity of an already-running vLLM server against the fixed wikitext-2
slice (scripts/build_perplexity_slice.py), via forced-decoding: the slice's
own token ids are sent as the prompt with max_tokens=0, echo=True,
prompt_logprobs=0, so the server never samples anything -- it just reports,
for each real token after the first, the log-probability it assigned to
that exact token given everything before it. No text is generated.

  NLL  = -mean(logprob of the actual next token), over all but the first
         token in the slice (the first has no preceding context to condition
         on -- vLLM reports it as null, excluded from the mean).
  PPL  = exp(NLL)

Confirmed against a live server before writing this script: prompt_logprobs=0
returns exactly {token_id: {logprob, rank, decoded_token}} per position, with
position 0 always null (see docs/decisions.md).

Repeated --repeats times against the same server, same token ids, no
sampling involved anywhere -- any spread across repeats is run-to-run
floating-point/kernel nondeterminism, not measurement noise from a random
process. That spread is the noise band this number must be read against.

Usage:
  python scripts/perplexity.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct --label fp16 \
      --slice data/wikitext2_test_slice_token_ids.json \
      --repeats 5 --out results/h200/perplexity_fp16.json
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
    p.add_argument("--slice", default="data/wikitext2_test_slice_token_ids.json")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default="results/perplexity.json")
    return p.parse_args()


def main():
    cfg = parse_args()

    with open(cfg.slice) as f:
        slice_meta = json.load(f)
    token_ids = slice_meta["token_ids"]

    per_repeat = []
    for i in range(cfg.repeats):
        result = run_once(cfg.base_url, cfg.model, token_ids, cfg.timeout)
        per_repeat.append(result)
        print(
            f"  repeat {i + 1}/{cfg.repeats}: "
            f"ppl={result['perplexity']:.4f} nll={result['nll']:.6f} "
            f"latency={result['latency_s']:.2f}s"
        )

    ppls = [r["perplexity"] for r in per_repeat]
    nlls = [r["nll"] for r in per_repeat]
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
    }

    payload = {
        "provenance": provenance.capture(
            model=cfg.model,
            vllm_server_url=cfg.base_url,
            extra={
                "script": "perplexity.py",
                "label": cfg.label,
                "slice_source_repo": slice_meta["source_repo"],
                "slice_source_file": slice_meta["source_file"],
                "slice_source_file_sha256": slice_meta["source_file_sha256"],
                "slice_tokenizer": slice_meta["tokenizer"],
                "slice_n_tokens": slice_meta["n_tokens"],
            },
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
