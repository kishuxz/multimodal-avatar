# Decisions

Non-obvious choices, what we rejected, and why. Written so I can answer
"why did you do it that way" out loud without re-deriving it.

## Repo scaffold (Phase 0)

**Chose:** plain Python repo, no web framework, no CI/lint/pre-commit yet.
**Rejected:** adding a test runner / linter up front.
**Why:** this is a benchmarking repo, not a service. The thing that needs
rigor is the measurement methodology, not the code style. CI can come
later if the repo grows scripts that need protecting from regressions;
right now every script is run manually and its output is inspected by
hand, which is a stronger check than a linter would give us.

## Provenance capture (`bench/provenance.py`)

**Chose:** ask the running vLLM server for its own version via `GET
/version`, falling back to a local `import vllm` only if no server URL is
given.
**Rejected:** always trusting a local `pip show vllm` / `import vllm`.
**Why:** the server runs inside a pinned Docker image; the host driving
the benchmark may have a different (or no) vLLM installed locally. Asking
the process that actually served the requests is the only way to avoid
silently recording the wrong version.

**Chose:** record two different "CUDA version" fields — the driver's
max-supported CUDA version (from `nvidia-smi`'s text header) and the
pinned vLLM Docker image tag, and call out in the field name that the
former is not ground truth.
**Rejected:** reporting a single `cuda_version` field.
**Why:** `nvidia-smi` reports the *maximum* CUDA version the installed
driver supports, not the CUDA runtime actually linked into the vLLM
container. Reporting one unqualified number would look precise but be
misleading. The Docker image tag is the authoritative source for what
actually ran; the driver number is just context (and a floor).

## Dependencies

**Chose:** `requirements.txt` starts empty; every addition needs explicit
sign-off before it's added.
**Why:** small and boring is a stated constraint. `bench/provenance.py`
needed zero third-party packages (subprocess + urllib cover it), so
nothing was added in Phase 0.

## Repo history (recreated from a single commit)

**Chose:** squash the entire Phase 0 history into one commit and start
this repo fresh from it, rather than keep the original multi-commit
history.
**Rejected:** rewriting the old repo's `main` branch in place.
**Why:** a squash-merged PR leaves its pre-squash commits reachable
through GitHub's own PR ref, independent of what's on `main` — rewriting
`main` alone doesn't remove them, and they'd surface the moment this repo
is ever made public. Recreating from one clean commit removes the
question entirely instead of relying on `main` staying tidy while earlier
history isn't. The original repo still exists, renamed and kept private,
purely as a backup.
**How to apply:** history hygiene is a first-commit decision, not a
cleanup pass. Every commit from here on is written as if the repo could
go public tomorrow.

## Pre-push attribution guard

**Chose:** a local `pre-push` hook that rejects a push if, across the
whole range being pushed, any commit's message (subject + body) matches
`claude|anthropic|co-authored-by: claude|generated with|🤖`
case-insensitive, **or** any commit's author/committer email isn't the
`135404520+kishuxz@users.noreply.github.com` identity.
**Rejected:** relying on remembering to run the manual `git log | grep`
check before every push. Also rejected: checking only the commit
message. The identity check was added after a squash-merge done through
the GitHub web UI, while logged into a different account, produced a
commit authored by that account's real work email — a message-only
check would never have caught this, since nothing in the message was
wrong.
**Why:** the manual check already got skipped once — a PR merged with an
AI co-author trailer before this repo was recreated, which is the reason
it needed recreating at all. A hook that runs automatically doesn't
depend on remembering. The identity half exists because local git
config only governs commits made locally; it has no effect on who
GitHub credits when a PR is merged through the browser under a
different logged-in account.
**How to apply:** hooks live under `.git/hooks` and are not versioned by
git, so this file (or a copy of it) does not travel with a fresh clone —
it has to be installed by hand. In a normal (non-worktree) clone, drop
the script at `.git/hooks/pre-push` and `chmod +x` it. In a repo checked
out via `git worktree`, hooks are shared across every worktree of that
repo by default (they live in the common `.git` directory, not per
worktree), so installing it once covers all worktrees — confirm with
`git rev-parse --git-path hooks` from each worktree if in doubt. The
hook only catches local pushes -- it cannot stop a bad identity from a
browser-driven merge, so the account logged into GitHub's web UI still
has to be the right one.
Verified by attempting a push with a deliberately bad commit message on
a throwaway branch (rejected, nothing reached the remote; a follow-up
push with a clean message succeeded), and separately by attempting a
push with a correct message but the wrong author/committer identity
(also rejected).

**Correction:** the identity check originally required committer ==
kishuxz unconditionally. That's wrong -- GitHub itself is always the
committer on a squash-merge (`noreply@github.com`), which is normal
merge mechanics, not an attribution problem. First push after PR #16
merged onto `main` was rejected by my own hook for exactly this reason.
Fixed to require author == kishuxz always, and committer == kishuxz OR
GitHub's own bot email. Re-verified both directions: the real merge
history now passes, a genuinely bad author/committer is still rejected.

## Barge-in timing (`harness.py`, issue #4)

**Chose:** run the SSE read as its own `asyncio.Task` and race it against
`asyncio.wait(..., timeout=abort_after)`. If the timer wins, cancel the
read task -- regardless of whether it had produced a token yet.
**Rejected:** the original design, which only checked for an abort inside
the token-received branch of the read loop. That meant an abort sampled
to happen before TTFT couldn't fire until the first token arrived, which
made it structurally impossible to represent a user talking over the
avatar before it had said anything -- the exact full-duplex case this
repo exists to measure.
**Rejected (alternative):** closing the response (`resp.close()`) from a
timer task and relying on that to raise inside the `async for` loop.
Cancelling the read task directly is a well-defined asyncio primitive;
hoping a specific exception type surfaces from a mid-stream close is not,
and that fragility isn't worth taking on in the same change. `resp.close()`
still isn't called explicitly anywhere -- the connection-level propagation
question is issue #1, deliberately kept separate.
**Why:** the timer is now the single mechanism for both "abort before any
token" and "abort mid-stream" -- it doesn't care which case it's in, so
there's one code path to reason about instead of two.
**How to apply:** `TurnResult.ttft` stays `None` when the abort beats the
first token, and is never backfilled. `summarize()` partitions kept
requests into `ok` (has a TTFT), `errs` (a real failure), and `no_ttft`
(no error, no TTFT -- aborted before first token, or, rarer, a genuine
zero-content-token completion). All three are counted explicitly in the
summary; none are silently excluded. Verified with
`--barge-in 1.0 --barge-in-min 0.001 --barge-in-max 0.005` against a local
mock SSE server with a deliberate 2s pre-token delay: 12/12 requests came
back with `ttft: null`, `abort_before_first_token: true`, and
`requests_ok: 0` while `requests_no_ttft` correctly carried all 12 --
nothing vanished from the summary. A second run with the abort window
moved past the first token, and a third with no abort at all, reproduced
the original TTFT/ITL/token-count behavior unchanged.

## Squash-merge authorship (what actually determines it)

**Found:** GitHub's squash-merge sets the resulting commit's author to
whoever opened the pull request, not whoever performs the merge and not
the branch commits' own author. Confirmed by ruling out the alternatives
directly, not by inference: same wrong author (`kishore-crux`, real work
email) came out of a browser-driven merge and a `gh pr merge --squash`
API call made while authenticated as `kishuxz` -- so "who clicks merge"
isn't it. Then a squash of a branch whose sole commit was independently
verified as `kishuxz` on both author and committer *still* produced
`kishore-crux` as the primary author, with `kishuxz` demoted to a
`Co-authored-by` trailer -- so "branch commit authorship" isn't it
either. Both PRs had been opened while `kishore-crux` was the active
account, before this repo's ownership and my authentication moved to
`kishuxz`; that's the one variable that lined up with both bad results.
**Why it matters:** the pre-push hook (identity + message checks) only
runs on local pushes. It cannot see or block a GitHub-side squash-merge,
so it gave no protection against this at all -- two more bad commits
landed on `main` after the hook already existed.
**What we do about it:** nothing new to build. Every PR from here on is
opened while authenticated as `kishuxz` (that's now the only account
signed in), which removes the one variable that caused this. If a PR
ever gets opened under the wrong account again, the fix is the same
rewrite-plus-force-push-with-lease used three times already in this
repo's short history -- verify the resulting merge commit's raw
`commit.author`/`commit.committer` immediately after every merge
(`gh api repos/.../commits --jq '.[0].commit'`), not the GitHub-resolved
`.author.login`, since that field is what actually lands in git history.

## Provenance wired into harness.py (issue #15)

**Chose:** `harness.py`'s `main()` calls `bench.provenance.capture()` and
writes the result as a `provenance` key in every output JSON, before
`config`/`summary`/`requests`.
**Why:** the helper existed since Phase 0 but nothing called it -- every
result written so far, including the Phase 1 sanity run, had zero
provenance. That's a direct violation of this repo's own first rule.
**Found while wiring it up:** `cfg.base_url` is `http://host:port/v1`
(the OpenAI API prefix), but vLLM's `/version` endpoint lives at the
server root. Passing `cfg.base_url` straight through would 404 and
silently fall back to a local `import vllm` -- defeating the documented
"ask the server, not the local install" design. Fixed by stripping the
path before calling `capture()`.

**Chose:** `bench/provenance.py` tries local git first (`git rev-parse
HEAD`); only if that fails does it read `PROVENANCE_GIT_SHA` /
`PROVENANCE_GIT_DIRTY` from the environment, and it always records which
source won as `git_sha_source`.
**Rejected:** trusting the env vars unconditionally, or leaving the SHA
null when `.git` is absent.
**Why:** `git archive` (how code gets onto the pod -- see the pod
environment entry above) strips `.git` entirely, so the pod can never
self-report a SHA. But a null SHA silently defeats the point of
provenance just as much as a wrong one would. The fix ships the SHA from
the machine that actually did the transfer (`git rev-parse HEAD` +
dirty-check, run locally, passed as env vars) and has the helper prefer
live local git whenever it's available -- so a real clone or a future
git-based deploy still self-reports correctly, and only the tree-export
case falls back to the shipped value. The source is always recorded
because a wrong SHA is worse than a null one: null is an obvious gap,
a stale or mismatched SHA looks trustworthy and isn't.

## deep_gemm import failure on the pod (non-fatal, flagged for later)

**Found:** at fp16 server startup, vLLM logs a caught, non-fatal warning:
`Module vllm.third_party.deep_gemm was found but failed to import` --
`ImportError: libnvrtc.so.13: cannot open shared object file: No such
file or directory`. The server starts and serves requests normally.
**Why it matters:** `deep_gemm` is a GEMM kernel library vLLM can use for
certain quantized/grouped matmul paths. It not being available doesn't
block fp16, but if the AWQ-Int4 or FP8 arms come back with surprising
numbers (slower than expected, or an unexpected kernel selected), this
is the first thing to check -- a missing accelerated kernel path would
silently fall back to a slower one rather than error, which is exactly
the kind of thing that would otherwise look like a real quantization
finding and isn't.
**Not fixed yet:** the missing library is a CUDA 13 runtime component
(`libnvrtc.so.13`) that isn't present despite `torch.version.cuda`
reporting `13.0` -- likely a partial/mismatched CUDA runtime install
alongside torch's bundled one. Filed here rather than chased down now
since it isn't blocking fp16 verification.

## Abort propagation, confirmed (issue #1)

**Prediction, stated before measuring:** vLLM's OpenAI-compatible server
is generally understood to detect a client disconnect during streaming
and cancel the underlying generation. Cancellation was expected to
work; the uncertain quantity was abort -> slot-free latency.

**Chose:** an explicit `resp.close()` in `harness.py`'s abort path,
replacing the implicit release-on-`async with`-exit used before.
**Why:** issue #1 was opened specifically because "did it actually
abort, or did the server just keep computing into a dropped socket"
was unverified. Closing explicitly removes any dependency on aiohttp's
default behavior for a partially-read response being what we assumed.

**Found:** the prediction held. `scripts/verify_abort.py` sends one
`max_tokens=512` request, aborts after only 10 tokens (max_tokens is
high enough, and the abort early enough, that a fast slot-free can't be
confused with coincidental natural completion -- at this model's decode
speed, 512 tokens takes over a second; the script estimates that
remaining time from the pre-abort inter-token latency and flags it if
the measured latency comes anywhere close), and polls `GET /metrics`
for `vllm:num_requests_running` throughout. Five trials, four at the
tightest achievable polling interval: `vllm:num_requests_running`
dropped from 1 to 0 within **2.2-4.1ms of the abort being issued**,
against an estimated 1.0-1.5s of remaining generation if it hadn't been
cancelled -- not ambiguous.

**Caveat, stated plainly:** the measured latency and the achieved
polling resolution are close to each other (observed median poll
interval ~3.1ms at `--poll-interval 0`; measured latencies 3.7-4.1ms
in that setting). That means the true cancellation latency is
somewhere between "the previous poll, which hadn't seen it yet" and
"this poll, which did" -- i.e., genuinely fast (low single-digit
milliseconds), but this measurement cannot resolve a more precise
number than that without a fundamentally different approach (e.g.
instrumenting the server directly rather than polling it externally).
Reporting "fast, single-digit ms, resolution-limited" rather than a
falsely precise point estimate.
**How to apply:** any future arm/config that changes abort behavior
(e.g. a different `max_tokens`, a different quantization backend) gets
re-verified with this script rather than assumed to inherit this
result -- it measures the mechanism, not a universal constant.

## Predictions, stated before measuring (Phase 3 sweep)

Qwen2.5-1.5B: 28 layers, 12 query heads, 2 KV heads -- aggressive GQA
(6:1), so the KV cache is roughly a sixth the size of full multi-head
attention at the same context length. On an 80GB H100 with a 1.5B
model, there is almost no memory pressure for quantization to relieve:
weights are a few GB, KV cache has enormous headroom even unquantized.

**Predicted:** small or negative latency deltas between arms at low
concurrency. Any quantization benefit should appear only at high
concurrency, where batch memory pressure (KV cache competing with
weights for the 90% GPU-memory-utilization budget, more sequences
resident at once) starts to matter.

**Predicted, AWQ specifically:** may be *slower* than fp16 at low load.
AWQ's int4 weights require a dequantization step per matmul; that's a
fixed compute cost that only pays for itself when you were bandwidth-
bound on weight loading in the first place. At low concurrency with
this small a model, bandwidth was never the bottleneck -- so the dequant
overhead is pure addition, not a trade.

**Predicted, FP8 specifically:** same shape as AWQ's prediction (small/
negative at low load, possible benefit only at high concurrency) --
plus a scope note that matters for interpreting whatever the numbers
say: vLLM's on-the-fly `fp8_per_tensor` is weight-only with a *static*
scale computed from the weights themselves, not the calibrated W8A8
(weights and activations both quantized, activation scales fit against
a calibration dataset) that most published FP8 checkpoints use.
Confirmed directly from vLLM 0.26.0's own engine startup log (not
assumed): `quantization_config=QuantizationConfigArgs(linear=QuantSpec(
weight=QuantKey(dtype=torch.float8_e4m3fn, scale=ScaleDesc(static=True,
...)), activation=None), moe=QuantSpec(..., activation=None))` --
`activation=None` on both `linear` and `moe` means activations stay in
bf16, only weights move to fp8. This arm tests a *different*
intervention than a calibrated FP8 checkpoint would, and the README
must say so wherever FP8 numbers appear -- these predictions, and any
results, are about weight-only dynamic-scale FP8, not W8A8.

**If the data contradicts any of this, that is the more useful result**
-- it means either the memory-pressure model above is wrong for this
model/hardware combination, or something about this specific serving
setup (batching behavior, kernel selection, the `deep_gemm` import
failure noted above) is doing something the simple story doesn't
predict. Either way it gets written up, not smoothed over.

## FP8 arm: which flag, and why (issue: don't assume the API)

**Chose:** `--quantization fp8_per_tensor`.
**Rejected:** a bare `--quantization fp8` (does not exist in vLLM
0.26.0 -- `_ONLINE_SHORTHANDS` in `vllm/config/quantization.py` has no
plain `"fp8"` key; this is a genuine API change from older vLLM
versions where that flag existed, not a typo). Also rejected
`fp8_per_channel` and `fp8_per_block`: viable alternatives (per-channel
or per-block weight scale granularity instead of per-tensor), but
`fp8_per_tensor` is the simplest, most-standard-sounding on-the-fly
recipe and there's no reason in this project's goals to need finer
scale granularity.
**Why it matters:** the three online shorthands differ only in scale
*granularity* (per-tensor / per-channel / per-block), not in whether
activations are quantized -- source inspection alone left that
genuinely ambiguous (a code comment on `fp8_per_channel` claims
"dynamic per-token activation" while the `QuantSpec` it constructs only
sets `weight`, not `activation`). Resolved by loading the model and
reading vLLM's own engine-init log rather than trusting either the
comment or the schema in isolation -- see the prediction entry above
for the exact confirmed config.
**How to apply:** if a future vLLM upgrade changes `_ONLINE_SHORTHANDS`
or this repo ever wants a genuinely calibrated W8A8 FP8 arm (a
pre-quantized checkpoint, not an on-the-fly flag), that's a different
experiment and needs its own entry here, not a silent swap of what
"the FP8 arm" means.

## Holding concurrency constant, not arrival rate (Phase 3 sweep)

**Chose:** derive each arm's arrival rate independently via Little's
Law to target the same offered concurrency (~1 / 8 / 32) across fp16,
AWQ, and FP8 -- not the same arrival rate.
**Rejected:** holding arrival rate (requests/sec) constant across arms
and letting realized concurrency vary by arm.
**Why:** service time differs by arm (that's the whole point of
testing quantization), so the two choices are different experiments.
Holding arrival rate constant would mean a faster arm is automatically
evaluated at *lower* concurrency than a slower one -- the axis this
sweep is trying to measure an effect along would also be shifting
underneath the comparison, conflating "this arm is faster" with "this
arm was tested at an easier operating point." Holding concurrency
constant instead means every arm is compared at the same point on the
load curve, which is also the standard framing for reporting
quantization results elsewhere (throughput/latency *at a given batch
size or concurrency*) -- and it's what the predictions above are stated
in terms of ("at high concurrency"), so it's the comparison that
actually tests them.
**How to apply:** "concurrency ~= 1/8/32" is the *offered* load implied
by a low-load (unqueued) service-time measurement via Little's Law, not
a guarantee of the realized average number of requests in flight --
under real queueing near saturation, realized concurrency can run
higher than the target once a server falls behind. The README names
this as the design that ran; a different, arrival-rate-held-constant
sweep would answer a different question and needs to be labeled as
such if it's ever run.

## --gpu-memory-utilization fixed at 0.9 across every arm

**Chose:** every server start in `scripts/sweep.sh` passes
`--gpu-memory-utilization 0.9`, unconditionally, regardless of arm.
**Why:** this setting determines how much VRAM is left for the KV
cache after weights and activation memory are reserved. Letting it
vary per arm (e.g. raising it for a smaller-weight quantized arm to
"use the freed memory") would mean any latency/throughput difference
observed is partly attributable to a different KV cache budget, not
just the quantization -- exactly the confound the Phase 1 harness
critique flagged as a way to accidentally measure a setting instead of
an intervention. `scripts/sweep.sh`'s `confirm_cache_cold` check also
verifies the prefix cache is actually empty right after each restart,
rather than assuming a fresh process implies a cold cache.

## Prefix caching: off is the cross-arm baseline (caught before it shipped)

**Found:** the first calibration pass ran fp16 with prefix caching on
and AWQ/FP8 with it on too by default in the original `sweep.sh` --
but that default was never actually deliberate, it was just what
`--enable-prefix-caching` happened to be wired to for every non-fp16
arm. The result: fp16's calibrated mean service time (60.7ms) looked
faster than AWQ's (68.5ms), and there was no way to tell how much of
that gap was quantization versus prefix caching, since both arms had
it on. On a multi-turn workload built from eight fixed conversations
(`harness.py`'s `USER_TURNS`), prefix caching is not a minor effect --
repeated prefixes are exactly what it's designed to exploit.
**Chose:** prefix caching off is now the baseline for every arm.
AWQ and FP8 run with it off, full stop. fp16 runs both off (the same
baseline as the other two arms) and on (its own additional dimension,
not a different starting condition).
**Why:** the whole point of comparing arms is that everything except
the quantization is held equal. A cross-arm latency gap is only
attributable to quantization if every other lever, including prefix
caching, was in the same position for every arm. fp16-on-vs-off stays
a legitimate, separate comparison -- it just isn't the number that
gets compared against AWQ or FP8.
**How to apply:** any future arm added to this sweep runs with prefix
caching off by default, matching the baseline, unless prefix caching
itself is the thing being studied for that arm.
