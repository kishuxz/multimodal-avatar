"""
Verify that vLLM actually cancels generation when the client disconnects
mid-stream, and measure how long that takes.

Prediction, stated before this script is run against real data: vLLM's
OpenAI-compatible server is generally understood to detect a client
disconnect during streaming and cancel the underlying generation.
Cancellation is expected to work. The uncertain, interesting quantity is
abort -> slot-free latency -- how long between the client disconnecting
and the server actually freeing the KV-cache slot
(vllm:num_requests_running dropping back to baseline). If the data
contradicts the prediction, that is the more valuable outcome, and the
result JSON says so plainly rather than being worked around.

Method: send one request with a high max_tokens (so it is unambiguously
still running when we abort -- see the max_tokens note below), read a
small, fixed number of tokens so generation is confirmed in progress,
then explicitly close the connection (mirroring harness.py's abort path)
and poll GET /metrics for vllm:num_requests_running before, during, and
after, at the fastest interval this client can sustain against
localhost. The actual observed polling interval is recorded alongside
the latency number -- a latency finer than the polling resolution is not
a measurement, it's a description of how fast we polled.

Why abort after only a few tokens rather than partway through: at this
model's observed decode speed, max_tokens=512 completes naturally in
roughly a second. If the abort fired near the end of that second, a fast
slot-free could just be coincidental natural completion, not evidence of
real cancellation. Aborting a few tokens in leaves most of a second of
separation between "cancelled early" and "would have finished anyway" --
this script estimates that remaining-natural-completion time from the
pre-abort inter-token latency and reports it alongside the measured
latency so the two can't be confused.

Usage:
  python scripts/verify_abort.py --base-url http://localhost:8000 \
      --model Qwen/Qwen2.5-1.5B-Instruct --max-tokens 512 \
      --abort-after-tokens 10 --out results/verify_abort.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import aiohttp

# Scripts in scripts/ run with their own directory as sys.path[0], not the
# repo root -- unlike harness.py, which lives at the root and gets this for
# free. Add the root explicitly so `bench` resolves regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance

METRIC_RE = re.compile(r'vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)')

PROMPT = (
    "Write a long, detailed story about a journey across a continent, "
    "with several characters, many events, and vivid description. Do not "
    "stop early -- keep going for as long as you can."
)


async def _read_running_metric(session, metrics_url):
    async with session.get(metrics_url) as resp:
        text = await resp.text()
    m = METRIC_RE.search(text)
    return float(m.group(1)) if m else None


async def _poll_metrics(session, metrics_url, samples, stop_event, interval_s):
    while not stop_event.is_set():
        t = time.perf_counter()
        value = await _read_running_metric(session, metrics_url)
        samples.append((t, value))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def run_trial(base_url, model, max_tokens, abort_after_tokens,
                     poll_interval_s, poll_settle_s):
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    samples = []
    stop_event = asyncio.Event()
    itls = []

    timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        baseline_running = await _read_running_metric(session, metrics_url)

        poller = asyncio.create_task(
            _poll_metrics(session, metrics_url, samples, stop_event, poll_interval_s)
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        }

        n_tokens = 0
        last_tok_t = None
        abort_issued_at = None
        request_sent_at = time.perf_counter()

        async with session.post(chat_url, json=payload) as resp:
            if resp.status != 200:
                stop_event.set()
                await poller
                raise RuntimeError(f"http {resp.status}: {await resp.text()}")

            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                text = delta.get("content")
                if not text:
                    continue

                now = time.perf_counter()
                if last_tok_t is not None:
                    itls.append(now - last_tok_t)
                last_tok_t = now
                n_tokens += 1

                if n_tokens >= abort_after_tokens:
                    abort_issued_at = time.perf_counter()
                    resp.close()
                    break

        # Keep polling after the abort so the slot-free moment, if any, is
        # actually captured rather than missed by stopping too early.
        await asyncio.sleep(poll_settle_s)
        stop_event.set()
        await poller

    return {
        "baseline_running": baseline_running,
        "request_sent_at": request_sent_at,
        "abort_issued_at": abort_issued_at,
        "n_tokens_before_abort": n_tokens,
        "max_tokens": max_tokens,
        "pre_abort_itls_s": itls,
        "samples": [{"t": t, "running": v} for t, v in samples],
    }


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int((len(xs) - 1) * p / 100)
    return xs[k]


def analyze(trial):
    samples = [(s["t"], s["running"]) for s in trial["samples"]]
    baseline = trial["baseline_running"]
    abort_at = trial["abort_issued_at"]

    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    polling_resolution = {
        "n_samples": len(samples),
        "observed_median_interval_s": _pct(gaps, 50),
        "observed_p95_interval_s": _pct(gaps, 95),
    }

    pre_abort = [(t, v) for t, v in samples if abort_at is not None and t < abort_at]
    confirmed_running_before_abort = any(
        v is not None and baseline is not None and v > baseline for _, v in pre_abort
    )

    slot_freed_at = None
    if abort_at is not None and baseline is not None:
        for t, v in samples:
            if t >= abort_at and v is not None and v <= baseline:
                slot_freed_at = t
                break

    latency_s = (
        (slot_freed_at - abort_at)
        if slot_freed_at is not None and abort_at is not None
        else None
    )

    itls = trial["pre_abort_itls_s"]
    remaining_tokens = trial["max_tokens"] - trial["n_tokens_before_abort"]
    avg_itl = sum(itls) / len(itls) if itls else None
    estimated_remaining_natural_completion_s = (
        avg_itl * remaining_tokens if avg_itl is not None else None
    )

    # A latency that's most of the way to how long the request would have
    # taken to finish on its own isn't distinguishable from "it just
    # finished" -- flag that rather than silently reporting a number.
    ambiguous_vs_natural_completion = (
        latency_s is not None
        and estimated_remaining_natural_completion_s is not None
        and latency_s > 0.5 * estimated_remaining_natural_completion_s
    )

    return {
        "confirmed_running_before_abort": confirmed_running_before_abort,
        "abort_to_slot_free_latency_s": latency_s,
        "cancellation_confirmed": latency_s is not None,
        "slot_freed_at": slot_freed_at,
        "estimated_remaining_natural_completion_s": estimated_remaining_natural_completion_s,
        "ambiguous_vs_natural_completion": ambiguous_vs_natural_completion,
        "polling_resolution": polling_resolution,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--abort-after-tokens", type=int, default=10)
    p.add_argument("--poll-interval", type=float, default=0.005,
                   help="requested seconds between /metrics polls; see "
                        "polling_resolution in the output for what was "
                        "actually achieved")
    p.add_argument("--poll-settle", type=float, default=5.0,
                   help="seconds to keep polling after the abort")
    p.add_argument("--out", default="results/verify_abort.json")
    return p.parse_args()


async def main():
    cfg = parse_args()

    trial = await run_trial(
        cfg.base_url, cfg.model, cfg.max_tokens, cfg.abort_after_tokens,
        cfg.poll_interval, cfg.poll_settle,
    )
    findings = analyze(trial)

    payload = {
        "provenance": provenance.capture(
            model=cfg.model,
            vllm_server_url=cfg.base_url,
            extra={"script": "verify_abort.py", "cli_args": vars(cfg)},
        ),
        "config": vars(cfg),
        "findings": findings,
        "trial": trial,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(findings, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    asyncio.run(main())
