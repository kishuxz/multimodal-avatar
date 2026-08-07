"""
Conversational-turn latency harness for a vLLM OpenAI-compatible server.

Measures, per request:
  - TTFT  : time to first token (arrival -> first content chunk)
  - ITL   : inter-token latency, per token after the first
  - E2E   : arrival -> final chunk (or abort)

Two load modes:
  - open   : Poisson arrivals at a fixed rate. Queueing delay shows up in TTFT,
             which is what you want. This is the mode that produces honest p99.
  - closed : N workers, each starting a new turn only after its last one
             finished. Self-throttling, so it HIDES overload. Reported for
             comparison only.

Barge-in: a configurable fraction of requests are aborted after a sampled
delay, simulating a user talking over the avatar -- possibly before the
avatar has said anything yet, not just mid-stream. We record when the abort
was issued, whether any token had arrived first, and measure the effect on
OTHER in-flight requests.

Usage:
  python harness.py --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --mode open --rate 8 --duration 120 --barge-in 0.25 \
      --out results/fp16_open_r8.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import statistics
import time
import urllib.parse
from dataclasses import dataclass, field, asdict

import aiohttp

from bench import provenance


# --------------------------------------------------------------------------
# Workload: multi-turn conversation, short prompts, short answers.
# Context grows turn over turn -- that growth is what pressures the KV cache.
# --------------------------------------------------------------------------

USER_TURNS = [
    "Hey, can you hear me okay?",
    "I've been thinking about something all week and I can't shake it.",
    "It's about my job. I'm not sure I should stay.",
    "The work is fine. It's the people I'd be leaving.",
    "Do you think that's a bad reason to stay somewhere?",
    "What would you do?",
    "Yeah. I think I already knew that.",
    "Can we talk about something lighter for a minute?",
]

SYSTEM_PROMPT = (
    "You are a warm, emotionally present conversational companion in a live "
    "voice call. Reply in one or two short spoken sentences. Never use lists "
    "or markdown."
)


def build_conversation(turn_index: int) -> list[dict]:
    """Conversation history up to and including user turn `turn_index`."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(turn_index):
        msgs.append({"role": "user", "content": USER_TURNS[i]})
        # Stand-in assistant history. Fixed text keeps context length
        # deterministic across arms, so KV-cache pressure is comparable.
        msgs.append(
            {
                "role": "assistant",
                "content": "I hear you. That sounds like it's been sitting "
                "with you for a while.",
            }
        )
    msgs.append({"role": "user", "content": USER_TURNS[turn_index]})
    return msgs


# --------------------------------------------------------------------------
# Per-request record
# --------------------------------------------------------------------------

@dataclass
class TurnResult:
    req_id: int
    turn_index: int
    prompt_chars: int
    arrival: float                 # absolute wallclock, for overlap analysis
    ttft: float | None = None      # seconds
    e2e: float | None = None       # seconds
    itls: list[float] = field(default_factory=list)
    n_tokens: int = 0
    aborted: bool = False
    abort_at: float | None = None  # absolute wallclock the abort was issued
    abort_before_first_token: bool = False
    error: str | None = None


async def _read_stream(resp: aiohttp.ClientResponse, res: TurnResult, t0: float) -> None:
    """Consume the SSE stream, updating `res` in place as tokens arrive.

    Runs as its own task so a barge-in timer can cancel it without caring
    whether any token has arrived yet -- cancellation is a well-defined
    asyncio primitive, unlike hoping a mid-stream `resp.close()` surfaces a
    specific exception type from inside an `async for`.
    """
    last_tok_t: float | None = None

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
        if res.ttft is None:
            res.ttft = now - t0
        else:
            res.itls.append(now - last_tok_t)
        last_tok_t = now
        res.n_tokens += 1


async def run_turn(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req_id: int,
    turn_index: int,
    max_tokens: int,
    barge_in_prob: float,
    barge_in_delay: tuple[float, float],
    rng: random.Random,
) -> TurnResult:
    msgs = build_conversation(turn_index)
    prompt_chars = sum(len(m["content"]) for m in msgs)

    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        # Ask the server to report usage so we can sanity-check our own count.
        "stream_options": {"include_usage": True},
    }

    will_abort = rng.random() < barge_in_prob
    abort_after = rng.uniform(*barge_in_delay) if will_abort else None

    t0 = time.perf_counter()
    res = TurnResult(
        req_id=req_id,
        turn_index=turn_index,
        prompt_chars=prompt_chars,
        arrival=t0,
    )

    try:
        async with session.post(
            f"{base_url}/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                res.error = f"http {resp.status}: {(await resp.text())[:200]}"
                return res

            read_task = asyncio.create_task(_read_stream(resp, res, t0))

            if abort_after is None:
                await read_task
            else:
                # Barge-in: race the read against a wall-clock timer that
                # knows nothing about token arrival. If the timer wins, the
                # user "spoke over" the avatar -- whether or not it had
                # said anything yet.
                _, pending = await asyncio.wait({read_task}, timeout=abort_after)
                if read_task in pending:
                    res.aborted = True
                    res.abort_at = time.perf_counter()
                    res.abort_before_first_token = res.ttft is None
                    read_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, aiohttp.ClientError):
                        await read_task
                else:
                    await read_task

    except asyncio.TimeoutError:
        res.error = "timeout"
    except aiohttp.ClientError as e:
        res.error = f"client: {type(e).__name__}"

    res.e2e = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------
# Load modes
# --------------------------------------------------------------------------

async def open_loop(cfg, session, rng) -> list[TurnResult]:
    """Poisson arrivals. Requests are launched on schedule regardless of
    whether earlier ones have finished -- so the server can fall behind, and
    that shows up in the tail. This is the honest mode."""
    tasks: list[asyncio.Task] = []
    start = time.perf_counter()
    req_id = 0

    while (time.perf_counter() - start) < cfg.duration:
        tasks.append(
            asyncio.create_task(
                run_turn(
                    session, cfg.base_url, cfg.model, req_id,
                    rng.randrange(len(USER_TURNS)), cfg.max_tokens,
                    cfg.barge_in, (cfg.barge_in_min, cfg.barge_in_max), rng,
                )
            )
        )
        req_id += 1
        # Exponential inter-arrival -> Poisson process.
        await asyncio.sleep(rng.expovariate(cfg.rate))

    return list(await asyncio.gather(*tasks))


async def closed_loop(cfg, session, rng) -> list[TurnResult]:
    """N workers in lockstep. Included only as a contrast: it cannot overload
    the server, so its p99 will look better than reality."""
    results: list[TurnResult] = []
    start = time.perf_counter()
    counter = {"n": 0}

    async def worker():
        while (time.perf_counter() - start) < cfg.duration:
            rid = counter["n"]
            counter["n"] += 1
            results.append(
                await run_turn(
                    session, cfg.base_url, cfg.model, rid,
                    rng.randrange(len(USER_TURNS)), cfg.max_tokens,
                    cfg.barge_in, (cfg.barge_in_min, cfg.barge_in_max), rng,
                )
            )

    await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results: list[TurnResult], warmup: int, wall: float) -> dict:
    # Discard warmup requests: the first few include CUDA graph capture,
    # allocator growth, and a cold prefix cache. Keeping them poisons p50.
    kept = results[warmup:]
    errs = [r for r in kept if r.error is not None]
    no_err = [r for r in kept if r.error is None]

    # Requests with no observed TTFT: aborted before any token arrived, or
    # (rarer) a completion that produced zero content tokens on its own.
    # Neither belongs in the TTFT/ITL distributions -- there's no first-
    # token time to report -- but they must still be counted somewhere, or
    # the percentiles quietly get flattered by dropping exactly the
    # requests a barge-in-heavy arm interrupts the fastest.
    no_ttft = [r for r in no_err if r.ttft is None]
    ok = [r for r in no_err if r.ttft is not None]

    ttfts = [r.ttft for r in ok]
    itls = [x for r in ok for x in r.itls]
    toks = sum(r.n_tokens for r in ok)

    return {
        "requests_total": len(results),
        "requests_after_warmup": len(kept),
        "requests_ok": len(ok),
        "requests_error": len(errs),
        "error_kinds": sorted({r.error.split(":")[0] for r in errs}),
        "requests_no_ttft": len(no_ttft),
        "requests_aborted_before_first_token": sum(
            1 for r in no_ttft if r.abort_before_first_token
        ),
        "requests_zero_token_no_abort": sum(
            1 for r in no_ttft if not r.abort_before_first_token
        ),
        "aborted_total": sum(1 for r in kept if r.aborted),
        "wall_seconds": round(wall, 3),
        "output_tokens_total": toks,
        "output_tokens_per_sec": round(toks / wall, 2) if wall else None,
        "ttft_s": {
            "p50": pct(ttfts, 50), "p95": pct(ttfts, 95),
            "p99": pct(ttfts, 99), "mean": statistics.fmean(ttfts) if ttfts else None,
        },
        "itl_s": {
            "p50": pct(itls, 50), "p95": pct(itls, 95), "p99": pct(itls, 99),
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["open", "closed"], default="open")
    p.add_argument("--rate", type=float, default=8.0,
                   help="open mode: arrivals per second")
    p.add_argument("--concurrency", type=int, default=8,
                   help="closed mode: number of workers")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--warmup", type=int, default=20,
                   help="requests discarded before stats")
    p.add_argument("--barge-in", type=float, default=0.0,
                   help="fraction of requests aborted mid-stream, 0..1")
    p.add_argument("--barge-in-min", type=float, default=0.3)
    p.add_argument("--barge-in-max", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="", help="free-text arm name for the report")
    p.add_argument("--out", default="results/run.json")
    return p.parse_args()


async def main():
    cfg = parse_args()
    rng = random.Random(cfg.seed)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
    conn = aiohttp.TCPConnector(limit=0)  # do not let the client be the bottleneck

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0 = time.perf_counter()
        if cfg.mode == "open":
            results = await open_loop(cfg, session, rng)
        else:
            results = await closed_loop(cfg, session, rng)
        wall = time.perf_counter() - t0

    summary = summarize(results, cfg.warmup, wall)
    # provenance.capture() hits GET /version, which lives at the server
    # root -- strip the /v1 API prefix, or it 404s and silently falls back
    # to a local `import vllm` instead of asking the server that actually
    # ran the requests.
    parsed_base = urllib.parse.urlsplit(cfg.base_url)
    vllm_server_root = f"{parsed_base.scheme}://{parsed_base.netloc}"
    payload = {
        "provenance": provenance.capture(
            model=cfg.model,
            vllm_server_url=vllm_server_root,
            extra={"harness_label": cfg.label},
        ),
        "config": vars(cfg),
        "summary": summary,
        "requests": [asdict(r) for r in results],
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    asyncio.run(main())
