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

## Load-run duration: 20s, validated, not assumed

**Chose:** 20s per load run (was 120s in the original design), applied
uniformly across every load point in the sweep.
**Why:** at the concurrency~=32 rate (~500 req/s for this model),
120s is 60,000+ requests -- dominates GPU spend for a percentile
estimate that doesn't need that many samples. Checked rather than
guessed: ran the same config (fp16, prefix off, rate=513 req/s) at
20s and 60s. TTFT p99: 106.9ms (n=7970) vs 99.5ms (n=23831) -- about
7% apart, judged stable enough to use the shorter duration.
**Known tradeoff, disclosed rather than hidden:** applying 20s
*uniformly* means the low end of the matrix (concurrency~=1, roughly
16 req/s) gets only ~300 samples in a 20s window, versus ~1900 at the
original 120s. p50/p95 are still reasonably estimated at that sample
size; p99 there is noisier than at the high-concurrency points, where
the high arrival rate itself supplies plenty of samples regardless of
duration. This is a deliberate choice to control GPU cost, not an
oversight -- if the low-concurrency p99 turns out to matter for a
specific finding, that arm/load point can be re-run at longer duration
individually rather than paying the cost everywhere.
**How to apply:** `CALIB_DURATION` (30s) is unchanged -- calibration
runs closed-loop at concurrency=1, which is inherently low-volume
(~500 requests in 30s) regardless of the eventual load rate, so it was
never the cost driver this check was about.

## Corrected results: noise band first, then the actual finding

The first write-up of these numbers (PR #18, before this entry) made
two mistakes worth recording alongside the fix, not just silently
correcting: it called FP8 "slower than fp16 at both ends" when the
data already showed it faster at low load and slower at high, and it
called a 2.4% single-run gap between AWQ and fp16 a "crossover"
without ever checking whether that gap was bigger than one arm's own
run-to-run noise. Both are now fixed, from real repeat data --
5 runs per arm, concurrency ~= 1 and ~= 32, prefix caching off,
identical config, seeds 0-4 (`scripts/repeat_check.py`,
`results/repeat_*.json`).

**TTFT p50 / p99, mean +/- stdev across 5 repeats:**

| | concurrency ~= 1 (p50) | concurrency ~= 1 (p99) | concurrency ~= 32 (p50) | concurrency ~= 32 (p99) |
|---|---|---|---|---|
| fp16 | 8.39 +/- 0.05 ms | 14.00 +/- 1.77 ms | 37.31 +/- 2.43 ms | 88.58 +/- 7.31 ms |
| AWQ | 10.47 +/- 0.37 ms | 16.85 +/- 1.23 ms | 37.79 +/- 0.76 ms | 91.35 +/- 9.21 ms |
| FP8 (weight-only) | 8.23 +/- 0.11 ms | 13.82 +/- 2.48 ms | 42.43 +/- 2.05 ms | 102.30 +/- 3.58 ms |

**AWQ vs fp16:** at concurrency ~= 1, AWQ is 2.08ms slower than fp16,
against a combined noise scale of a few tenths of a millisecond --
five-plus standard deviations, clearly real, matches the prediction
(dequant overhead, never bandwidth-bound at this batch size). At
concurrency ~= 32, AWQ's mean (37.79ms) is actually *higher* than
fp16's (37.31ms) -- the opposite direction from the single-run
"crossover" claim -- but the 0.48ms gap is well inside fp16's own
+/-2.43ms run-to-run noise. **Corrected finding: no measurable
difference between fp16 and AWQ at concurrency 32. At low load fp16
is clearly ahead.** There is no crossover in this data. A single run
each made it look like one; five did not.

**FP8 (weight-only) vs fp16 -- a real, opposite-direction pattern,
not the same story as AWQ:** at concurrency ~= 1, FP8 is marginally
faster (8.23 vs 8.39ms) -- small, on the edge of FP8's own noise
(+/-0.11ms), directionally consistent across all 5 repeats but not a
large effect. At concurrency ~= 32, FP8 is clearly slower (42.43 vs
37.31ms, ~14% higher) -- a ~5ms gap against a combined noise scale of
2-2.5ms on each side, a real difference, not noise.
**Candidate mechanism, stated as a hypothesis, not confirmed by
profiling:** weight-only FP8 halves the bytes moved per forward pass
(fp8 vs bf16 weights). At concurrency ~= 1, decode is memory-bandwidth
bound and weights are read from HBM once per forward pass regardless
of batch size -- smaller weights are a direct, nearly-free latency
win when there's little compute happening anyway. At concurrency ~=
32, the batch is large enough that the workload shifts compute-bound;
weight-loading cost is now amortized across many more tokens per
read, shrinking the bandwidth benefit, while the fp8-to-bf16 dequant
step needed before each matmul (activations stay bf16 -- see the FP8
flag entry above) is a fixed compute tax that does *not* shrink with
batch size, and now lands on top of an already-saturated GPU. Net:
a real slowdown once compute, not memory, is the bottleneck.
**Why AWQ doesn't show the same divergence** is a real open question,
not resolved here: one plausible reason is that vLLM's AWQ (int4)
kernels are far more mature/optimized than the generic
`fp8_per_tensor` weight-only path (AWQ is a widely-deployed, heavily
optimized method; this specific FP8 shorthand is a newer, simpler
recipe) -- if the FP8 dequant kernel is less fused/efficient, its
fixed overhead could scale worse with batch size than AWQ's. This is
a hypothesis, not a measured claim; confirming it would need a kernel-
level profile (e.g. Nsight), which is out of scope here.

**What this sweep's largest, most robust effect actually is --
restructuring the story around it:** prefix caching, not
quantization. fp16 with prefix caching on vs off at concurrency ~= 32
(single runs, not yet repeat-validated the way the quantization
comparison above is): TTFT p50 26.95ms vs 37.36ms (28% lower with
caching on), p99 55.06ms vs 99.65ms (45% lower). Both gaps are many
times larger than any quantization effect measured here, including
the real ones (FP8's ~14% slowdown at high load). On a multi-turn
conversational workload built from a small number of repeated
prefixes, prefix caching is the first-order lever; quantization is
second-order at this model size and these load levels. The README
leads with this, not with quantization.
**Not yet repeat-validated:** the prefix-caching effect size above is
from the original single-run sweep, same limitation the AWQ crossover
claim had before this correction. It's such a large gap (28/45%)
relative to any noise band observed so far (a few percent) that it's
very unlikely to be noise, but it hasn't been checked the same way
the quantization comparison now has. Worth a repeat pass before
leaning on the exact percentages in a final write-up, even though the
direction and rough magnitude are not in doubt.

**Scope of every claim above:** Qwen2.5-1.5B-Instruct, single H100
80GB, concurrency approximately 1 and approximately 32 (not the full
0-32 range), prefix caching off unless stated otherwise, vLLM 0.26.0.
None of this generalizes to other model sizes, other hardware, or
load levels between or beyond the two tested here without saying so.

## First real dependency: matplotlib (Phase 5)

**Chose:** add `matplotlib==3.11.1` to `requirements.txt` -- the first
non-empty entry since Phase 0.
**Why:** `scripts/analyze.py` and `scripts/kv_cache_check.py` produce
plots, and there's no stdlib way to rasterize a chart. Every other
Phase 5 computation (percentiles, mean/stdev, the matrix build) stays
on `statistics` and `json`, same as every script before it -- this is
the one place a third-party package earns its place, not a general
loosening of the "small and boring" default from Phase 0.
**How to apply:** the next dependency still needs its own sign-off and
its own entry here, same as this one -- matplotlib landing doesn't
make future additions the default.

## KV cache arithmetic check: no committed value to check it against (Phase 5)

**Found:** `scripts/kv_cache_check.py` computes the expected KV cache
footprint from Qwen2.5-1.5B-Instruct's own config -- 28 layers x 2 KV
heads x 128 head_dim x 2 (K/V) x 2 bytes (bf16) = 28,672 bytes/token
(28.0 KiB/token exactly), 939,524,096 bytes (~0.94 GB) for one sequence
at the model's full 32,768-token context. That figure is an upper bound
on a fully-extended sequence, not a steady-state prediction: vLLM's
paged allocator reserves blocks per token actually generated, not per
maximum possible length, and every request in the Phase 3 sweep ran
`max_tokens=80` -- nowhere near 32,768.
**The check itself couldn't run as originally scoped:** the plan was to
compare this expected figure against what vLLM reported during the
Phase 3 sweep. Nothing in this repo has that number. `scripts/sweep.sh`
only ever polls `/metrics` for `vllm:kv_cache_usage_perc` (a percentage,
to confirm a cold cache before each run) -- never an absolute block or
byte count -- and the server's full startup log, where vLLM's own
`GPU KV cache size:` line actually lives, is written to
`/workspace/vllm_sweep_${label}.log` on the pod and never copied into
`results/`. `*.log` is also `.gitignore`'d. So there was never a
ground-truth figure committed to check the arithmetic against, on
either the sanity run or the full sweep.
**Why it matters:** this is a gap in what the harness captures, not a
disagreement in the numbers -- worth knowing before an interview either
way, and arguably more useful than a clean match would have been, since
a clean match wouldn't have surfaced that the sweep's own provenance is
incomplete. Filed as issue #19 (`measurement`) rather than fixed here,
scoped to the whole server startup log -- it carries the resolved
max-model-length, block size, scheduler settings, and attention backend
selection, not just the KV cache line, and none of that is captured
either right now.
**How to apply:** the next real GPU run should capture that log
(`scripts/sweep.sh` currently `tee`s it to the pod's local disk and
stops there) into a file committed alongside its `results/*.json`, so
this check has something to diff against and stops being one-sided.

## H200 environment rebuild: forced down to vLLM 0.19.1, not 0.26.0

**Found:** `pip install vllm` on the new pod (see the hardware-change
entry) resolves `vllm==0.26.0` with `torch==2.11.0+cu130` by default --
same as the H100 pod got. On the H100 that worked, because its driver
(`580.126.09`) supports CUDA 13. This pod's driver is `570.124.06`,
which `nvidia-smi` itself reports as capping out at `CUDA Version: 12.8`.
Starting the server failed immediately: `RuntimeError: The NVIDIA
driver on your system is too old (found version 12080)` -- torch's own
CUDA init refusing to run a CUDA-13-compiled runtime on a driver that
only certifies CUDA 12.8.
**Rejected:** forcing `torch==2.11.0+cu128` in afterward (works on its
own -- `torch.cuda.is_available()` returns `True`) while keeping
`vllm==0.26.0`. This does not fix it: vLLM's own compiled extension
(`vllm._C_stable_libtorch`) fails to import with `ImportError:
libcudart.so.13: cannot open shared object file`, regardless of what
torch build sits next to it. Checked PyPI directly (`pypi.org/pypi/
vllm/0.26.0/json`): there is exactly one Linux wheel per vLLM release,
no CUDA-version-specific variants the way torch has -- unlike torch,
picking a different index doesn't change which CUDA toolkit vLLM's own
binary was compiled against. Confirmed the same failure on
`vllm==0.25.1` paired with a matching cu128 torch -- its compiled
extension is also CUDA-13-linked, so this isn't specific to 0.26.0.
**Rejected:** trying to work around the missing library (it's still on
disk, provided by a separately-installed `nvidia-cu13` package, just
off vLLM's runtime search path after swapping torch) by re-exposing it
via `LD_LIBRARY_PATH`. Getting past the import would very likely just
move the failure to an actual CUDA driver-API call at request time,
probably a more confusing error -- CUDA 13.0's minimum driver
requirement is genuinely higher than this pod's 570-series driver
provides (NVIDIA's own compatibility table, not a vLLM-side check), so
there's no real fix here short of the driver itself changing, which is
outside what's controllable from inside the pod's container.
**Chose:** find the newest vLLM release still built against CUDA 12.x
and install that instead. Checked (cheaply, against PyPI's per-file
`.metadata` endpoint, no pod time spent) each release's declared
`torch==` pin going backward from 0.26.0: 0.22.1 through 0.26.0 all pin
`torch==2.11.0`; **0.19.1 (released 2026-04-18, three months before
0.26.0) is the newest release pinning `torch==2.10.0`.** Separately
confirmed torch's own default PyPI wheel is the actual version boundary
that matters: torch 2.8.0/2.9.0/2.9.1/2.10.0 all declare
`nvidia-cuda-runtime-cu12==12.8.90`; **2.11.0 is the first version whose
default wheel switched to `nvidia-cudnn-cu13`** -- i.e. this is a
PyTorch-level CUDA-13 cutover at 2.11.0, and vLLM inherited it at 0.20.0
by bumping its own torch pin. Installed `vllm==0.19.1` cleanly (no extra
index needed -- it naturally resolves `torch==2.10.0+cu128`) and
confirmed the server actually starts, loads the model, and serves a
request (`GET /v1/models` -> `200`) for all three arms:
fp16 (`Qwen/Qwen2.5-1.5B-Instruct`), FP8 (`--quantization fp8`), and AWQ
(`Qwen/Qwen2.5-1.5B-Instruct-AWQ`).
**Why it matters:** this is a seven-minor-version downgrade from the
H100 run's vLLM, forced by hardware/driver mismatch, not chosen for any
other reason. It has real downstream consequences, each checked
directly rather than assumed:
- **Attention backend changed.** H100 (vLLM 0.26.0) used
  `flashinfer-python==0.6.14`. H200 (vLLM 0.19.1) selects
  `FLASH_ATTN` (FlashAttention v3) instead: `Using FLASH_ATTN attention
  backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER',
  'TRITON_ATTN', 'FLEX_ATTENTION']`. `flashinfer-python` is still
  installed (`0.6.6`, an older version than the H100's `0.6.14`) but
  isn't the one vLLM picked here.
- **The FP8 flag changed, as `docs/decisions.md`'s own FP8-flag entry
  above predicted it might.** `--quantization fp8_per_tensor` (the H100
  flag) is not confirmed to exist in 0.19.1 -- `fp8_per_tensor` /
  `fp8_per_channel` / `fp8_per_block` don't appear in
  `vllm.model_executor.layers.quantization.QUANTIZATION_METHODS` for
  this version, but plain `'fp8'` does. `--quantization fp8` loads and
  serves successfully, selecting `CutlassFP8ScaledMMLinearKernel` for
  `Fp8OnlineLinearMethod` -- a different kernel path than the H100 arm.
  **Not yet verified:** whether this `fp8` method on 0.19.1 is
  weight-only/static-scale like the H100's `fp8_per_tensor` was, or a
  different recipe (calibrated W8A8, dynamic scale). That needs the same
  engine-log-reading investigation the original FP8-flag entry did
  before this FP8 arm can be trusted the way the H100's was --
  deliberately deferred to when the FP8 arm is actually run, not
  assumed here.
- **AWQ loads successfully**, auto-selecting `awq_marlin` ->
  `MacheteLinearKernel`. No H100-side record of which kernel it selected
  there, so this can't be directly compared yet, but it's now on record
  for whenever that comparison matters.
- **KV cache reporting is still capturable, which is good news for
  issue #19.** The startup log carries `Available KV cache memory:
  120.53 GiB` and `GPU KV cache size: 4,513,888 tokens` for the fp16
  arm at `--gpu-memory-utilization 0.9` -- format differs slightly from
  whatever 0.26.0 would have printed (untested, since 0.26.0 never got
  past CUDA init on this pod), but the substance issue #19 asked for is
  confirmed present in 0.19.1's log too.
- **`/version` and `/metrics`' `vllm:kv_cache_usage_perc` are unchanged**
  -- `bench/provenance.py` and `scripts/sweep.sh`'s `confirm_cache_cold`
  should work against this version without modification.
- `max_model_len` still resolves to `32768` (matches
  `scripts/kv_cache_check.py`'s assumption).
**Also found, unrelated to the CUDA mismatch:** this pod's base image
enforces PEP 668 (`externally-managed-environment`) where the H100
pod's apparently didn't (or the difference went unrecorded) --
`pip install vllm` needs `--break-system-packages` here.
**How to apply:** any Phase 3 re-run design has to route through vLLM
0.19.1's actual behavior, not assume 0.26.0's carries over. The FP8
weight/activation semantics specifically need re-confirming (same
method the original FP8-flag entry used: read the engine's own
`quantization_config` log line) before the FP8 arm's numbers can be
interpreted the way the H100 FP8 arm's were.

## Abort-window count spread by arm: exposure, not a different mechanism (Phase 5)

**Found:** `plots/itl_abort_windows.png` (concurrency ~= 32, barge-in
0.25) shows very different abort counts and window counts by arm: fp16
8 aborts / 8 windows, AWQ 33 aborts / 31 windows, FP8 19 aborts / 16
windows -- roughly a 4x spread between fp16 and AWQ. First instinct was
that this might mean barge-in itself behaves differently per arm, or a
data problem. It's neither.
**Why:** the sampled abort delay is a fixed `--barge-in-min`/`--barge-in-max`
(0.3-1.2s, uniform) applied identically to every arm -- it only actually
aborts anything if the delay elapses *before* the request would have
finished on its own. So the abort rate is really measuring "fraction of
requests still in flight past ~0.3s," which depends entirely on each
arm's own request duration at this load point. Checked directly against
the `bargein0.25` files themselves: fraction of (non-warmup, non-error)
requests with e2e > 300ms is 7.7% for fp16, 18.2% for AWQ, 13.2% for
FP8 -- the same ordering and roughly the same spread as the abort
counts (0.10% / 0.45% / 0.24%). AWQ being the slowest-tailed arm here
lines up exactly with AWQ having the highest ITL p50 among the three at
concurrency ~= 32 (8.57-8.80ms vs fp16's 5.05-5.33ms and FP8's
6.01-6.06ms -- see "Corrected results" above), which is the same AWQ-
decode-degradation-at-high-concurrency finding already recorded there,
now showing up a second way.
**Why it matters:** the with-abort-vs-without-abort *comparison within
each arm* (the actual plot) is unaffected by this -- it's still a valid
same-arm, same-run comparison. What it rules out is reading the window-
*count* itself as a cross-arm finding ("AWQ gets barge-in tested 4x as
often" is not a statement about barge-in; it's a restatement of AWQ
already being the slowest arm at this load point). The plot's caption
and `abort_window_itls()`'s docstring both say this now, so the number
isn't misread standalone.
**How to apply:** if barge-in exposure ever needs to be held constant
across arms for a real cross-arm comparison (not just within-arm, which
is what's reported here), the fix is deriving `--barge-in-min`/`--max`
per arm from that arm's own service-time distribution, the same way
`scripts/calibrate.py` already derives arrival rate per arm via Little's
Law -- not attempted here, out of scope for this pass.

## Sweep v2: four fixes folded into the design, not bolted on after

Before any GPU time goes toward a re-run, `scripts/sweep.sh` and
`scripts/analyze.py` get four changes -- each fixes something the H100
sweep either got wrong or only fixed after the fact, and each is
recorded here as a design change, not just a diff.

**1. Scaled barge-in window, not a fixed 0.3-1.2s.**
**Found:** in the H100 data, `aborted_total` is 0 for every concurrency
1 and 8 `bargein0.25` run -- the fixed window never once fired before
the request finished on its own at those load points. Only concurrency
32 (where queueing pushed response time up near the fixed window)
produced any aborts at all. The barge-in dimension of the entire H100
sweep was effectively tested at one load point out of three.
**Chose:** derive `--barge-in-min`/`--barge-in-max` per arm from that
arm's own calibrated mean service time (`scripts/calibrate.py`'s
concurrency=1 probe, already computed for the rate derivation) --
25% to 75% of it, computed once per arm and applied at all three
concurrencies. Early enough to reliably clear TTFT (a small fraction of
total service time here), late enough to leave a real gap before
natural completion.
**Rejected:** recalibrating the window per load point (i.e. accounting
for queueing-inflated response time at concurrency 32 specifically).
**Why:** a single per-arm scale, from the same unqueued measurement
`scripts/calibrate.py` already produces, is enough to fix the actual
bug -- zero aborts at two of three load points -- without adding a
second calibration pass. **Known limitation, stated rather than
hidden:** at concurrency 32 the window is sized off unqueued service
time, so it can land earlier in the request's actual (queued) lifetime
than "mid-decode" -- it will still reliably fire and interrupt
something in flight, just not necessarily at the same relative point in
the response every time. See `scripts/sweep.sh`'s own comment for the
exact fractions and reasoning.

**2. Server startup log captured per run, committed.**
**Found:** `scripts/sweep.sh` always `tee`'d each server's startup log
to the pod's local disk and never copied it anywhere this repo tracks
-- the KV-cache-size log line, resolved max model length, block size,
and attention backend selection all existed only for the duration of
that pod. This is what blocked `scripts/kv_cache_check.py` from having
anything to verify against (see that entry above) and is issue #19.
**Chose:** `start_server()` now copies the log to
`results/server_log_${label}.txt` immediately after the server is
confirmed ready and cold, before any load-test traffic -- a clean
startup-only snapshot, committed alongside that arm's `results/*.json`.
**Why now, not later:** confirmed directly (environment-rebuild PR) that
vLLM 0.19.1 still prints `Available KV cache memory:` and
`GPU KV cache size:` at startup, so this is a real capability, not a
hope.

**3. Repeats built into the matrix, not a separate manual pass.**
**Found:** the H100 run's noise-band numbers (`scripts/repeat_check.py`,
5 seeds at concurrency 1 and 32, prefix off, barge-in 0.0) came from a
second, manual invocation after the main sweep had already run and been
written up once with single-run numbers -- the original write-up
initially called a 2.4% single-run AWQ/fp16 gap a "crossover" before
the repeat pass showed it was noise (see "Corrected results" above).
That mistake was possible specifically because repeats weren't part of
the same pass that produced the single-run numbers in the first place.
**Chose:** `run_open_loop_matrix()` now calls `scripts/repeat_check.py`
directly, inline, for exactly the cells that were manually repeated
before (concurrency 1 and 32, prefix off, barge-in 0.0) -- same scope,
same script, just invoked automatically instead of as a follow-up step
someone has to remember to run. Non-repeat cells (concurrency 8,
barge-in 0.25, prefix-on) are unchanged, single-run.
**Rejected:** repeating every cell. 5x the GPU time for load points the
quantization comparison's noise band doesn't actually hinge on --
scope stays what it was, just automatic now.
**4. File-classification assertion in `scripts/analyze.py`.**
**Found:** `fp16_closed_c8.json` didn't match the run-file regex (its
name skips the `_pcoff`/`_pcon` token every other fp16 file has), so it
was silently absent from every table and plot the first time
`scripts/analyze.py` ran. Caught only because the closed-loop table
visibly had 2 of 3 arms and that looked wrong by eye -- nothing in the
script itself would have flagged it if the gap had been less visually
obvious (e.g. a missing single cell in a 60-row table).
**Chose:** `assert_full_classification()` globs every `results/*.json`
file and requires each one to either match a known pattern (run, repeat
seed, repeat summary, calibration) or appear in
`INTENTIONALLY_UNCLASSIFIED_PATTERNS` with a stated reason. Anything
left over raises, listing exactly which files, before any table or plot
gets built. Verified both directions: passes clean against the full
H100 `results/` directory (63 classified, 8 calibration files
explicitly excluded, 0 unaccounted), and fails loudly with the
offending filename when a deliberately unrecognized file is dropped in.
**Why:** the original bug was caught by luck (a visually obvious 2-of-3
gap). The next naming mismatch might not be as visible -- a single
missing repeat seed, a missing row in a 60-row table -- and shouldn't
need to be.
