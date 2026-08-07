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

**Correction, found during the Phase 7 public-readiness pass -- the
paragraph above was wrong about what the rewrite fixes:** "verify the
resulting merge commit... immediately after every merge" checks
whatever `main`'s tip looks like at that moment. It does not check, and
cannot fix, what a *pull request's own merge-commit record* shows --
those are two different things. Confirmed directly: `gh pr view <10|11|12>
--json mergeCommit` still returns `bfe2170`/`861f2ac`/`122662e` as each
PR's recorded merge commit, and `gh api repos/.../commits/<sha> --jq
'.commit.author'` on those exact SHAs still returns `kishore-crux
<kishore@livingforeverai.com>` -- live, on GitHub, right now, not a
stale local artifact. The rewrite-plus-force-push done three times
*did* work, and worked completely, for its actual target: `main`'s
current tip carries the correct identity all the way through, verified
the same way originally described. What it structurally cannot touch is
a PR's own merge-commit metadata, which GitHub treats as permanent the
moment a PR is merged -- rewriting the branch that fed it, or the
`main` it landed on, doesn't retroactively edit that record, and GitHub
provides no self-service way to. **Decision, not an oversight:** accept
this rather than recreate the repo a second time (the fix used for the
*original* pre-this-repo authorship problem, `docs/decisions.md`,
"Repo history recreated from a single commit"). Three PRs out of 34
carrying a former, identifiably-mine work account is a smaller cost
than discarding the visible process history this repo's whole
credibility argument rests on. If this repo is ever recreated for an
unrelated reason, this is resolved as a side effect; it does not justify
recreating the repo on its own.
**How to apply:** the pre-push hook and the "verify main's tip" habit
both still matter and both still work -- they're just answering "is
`main` clean," not "is every PR's own metadata clean." Those are
different guarantees; don't conflate them again.

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

**Re-verified on H200 (this entry's own rule, applied to itself):** the
numbers above are from the H100 run; this project went a full seven
sections (hardware change, vLLM downgrade, four sweep-v2 fixes) without
actually re-running this specific check on H200, despite "any future
arm/config that changes abort behavior gets re-verified" being the
stated rule immediately above. Fixed during the Phase 7 audit pass, not
assumed inherited. `results/h200/verify_abort_trial{0..4}.json` -- 5
trials, `--max-tokens 512 --abort-after-tokens 10`, fp16, vLLM 0.19.1:
`vllm:num_requests_running` dropped from 1 to 0 within **4.15-5.15ms**
of the abort being issued, `cancellation_confirmed: true` and
`ambiguous_vs_natural_completion: false` on all 5 trials. Still fast,
still single-digit ms -- the prediction holds on H200 too. **The
polling resolution itself differs from the H100 run** (median ~8.4ms
here vs. ~3.1ms there, both at `--poll-interval 0.005`) -- plausibly
network/scheduling variance between the two pods' `/metrics` round-trip
time, not investigated further since it doesn't change the conclusion
(the measured latency is still below the polling interval on both
environments, so both reports are "fast, resolution-limited," not two
different precise numbers to reconcile). `make verify-abort` wires this
into the Makefile going forward -- previously runnable but not listed
anywhere a reader would find it.

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
7% apart, judged stable enough to use the shorter duration. **Not
file-backed:** this was a one-off methodology check on the H100 pod, run
before the decision to commit every result JSON was applied this
consistently -- the 60s run's output was never saved, and the pod is
gone, so this specific comparison can't be reproduced or re-verified
the way the rest of this repo's numbers can. The 20s-vs-120s tradeoff
this check justifies is disclosed in the next paragraph regardless of
that gap, since the decision it informed (use 20s) is still in effect
and still worth explaining, but the 106.9ms/99.5ms figures themselves
should be read as "reported once, not independently checkable," not as
a number with the same evidentiary weight as everything else in this
repo.
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
**Superseded by a later baseline, found during the Phase 7 audit pass,
disclosed rather than silently reconciled:** the 45% p99 figure above
compares two single runs -- both sides unrepeated, self-consistent at
the time this was written. Repeats were later added for the *off* side
of this same cell (`results/repeat_fp16_c32.json`, 5 seeds, mean
88.58 +/- 7.31ms p99) without this section being revisited. Every
auto-generated artifact downstream of that (`results/summary.md`,
`plots/effect_size_comparison.png`, and the H200-section prose at
"AWQ crosses over," below) computes the *on* side (55.06ms, still
single-run -- prefix-caching-on was never repeat-validated on H100)
against that 88.58ms repeat-validated mean instead, giving **-37.8%**,
not -44.7%/45%. **-37.8%, against the repeat-validated mean, is the
canonical figure** -- it rests on 5x more data for the baseline side
and is what every other table in this repo already uses; this
section's own 45% is kept above as the historical single-run-vs-
single-run number it always was, not deleted, but is superseded, not
canonical.

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

## Hardware change: H100 80GB HBM3 -> H200 SXM 141GB (new pod)

**Found:** the pod behind every Phase 1-3 result (`results/*.json`, all
of `docs/decisions.md` above this entry, the whole Phase 5 write-up) was
stopped and its H100 reclaimed. The replacement pod is different
hardware, not a restart of the same one: `NVIDIA H200`, 143771 MiB
(~141GB, vs the H100's 80GB), driver `570.124.06` (vs `580.126.09`),
datacenter `US-NC-1` (vs `US-CA-2`), and a differently-provisioned
`/workspace` (volume disk, not the `avatar-bench-vol` network volume) --
confirmed empty on first login, so nothing carried over and the
environment gets rebuilt from scratch, not resumed.
**Why it matters:** every number in `results/*.json` and everything
derived from them (`results/summary.md`, `plots/*.png`,
`results/kv_cache_check.md`, and the corrected-findings section of this
file) is scoped to the H100 80GB HBM3 that produced it. GPU model,
memory capacity, and driver all differ on the new pod -- any of those
alone would be enough to invalidate a direct comparison; here all three
changed at once.
**Chose:** state this plainly, in both this file and the README, before
any new measurement runs -- not as a footnote on a table, as its own
visible fact. **No table in this repo mixes H100 and H200 numbers.**
Phase 1-3 gets fully re-run on the H200 (see the prediction entry
immediately below for what's expected to change and why) rather than
patched or extended in place.
**How to apply:** any future hardware change gets the same treatment --
recorded here before remeasuring, stated in the README's Setup section,
and no silent mixing of tables across the boundary. If a reader only
skims one number from this repo, which GPU produced it should never be
something they have to dig for.

## Predictions, stated before measuring (Phase 3 re-run on H200)

The H100 run's central story (see "Predictions, stated before measuring
(Phase 3 sweep)" above) was that Qwen2.5-1.5B on an 80GB card leaves
almost no memory pressure for quantization to relieve -- weights are a
few GB, KV cache has enormous headroom even unquantized, so any
quantization benefit should show up only at high concurrency, if at
all. The H200 has 140.4GB (143771 MiB reported by `nvidia-smi`), ~1.76x
the H100's 80GB. That doesn't change the story; it makes the same
argument stronger.

**Predicted:** at `--gpu-memory-utilization 0.9`, the KV cache budget
goes from ~72GB (H100) to ~126GB (H200) -- headroom the 1.5B model was
already not using up on the H100. **AWQ and FP8 should have even less
memory pressure to relieve on H200 than they did on H100**, not more --
if anything, whatever high-concurrency benefit either arm showed on the
H100 (FP8's real ~14% win at c~=32, in the direction bandwidth-bound
weight-loading predicts) should show up at a *higher* concurrency on
H200 than it did on H100, if it shows up in the 1/8/32 range tested
here at all. A same-or-larger benefit at c~=32 would be the surprise,
not the null result.

**Predicted, AWQ specifically:** the dequant step (int4 -> bf16 per
matmul) is a compute cost, not a memory-bandwidth cost. H200 is the
same Hopper compute architecture as H100 SXM -- its headline change is
memory (more capacity, ~1.4x more HBM bandwidth: HBM3e vs HBM3), not
more FLOPs. So AWQ's fixed dequant overhead should cost roughly the
same *absolute* time on H200 as it did on H100 (8.39ms fp16 vs 10.47ms
AWQ, both at c~=1). If everything else gets faster from the extra
bandwidth, that fixed cost becomes a *larger share* of AWQ's own total
latency, not a smaller one -- AWQ's relative low-load penalty should
hold or widen, not shrink.

**Predicted, FP8 specifically:** the low-load FP8 win on H100 (8.23 vs
8.39ms, marginal) was attributed to halving the bytes moved per forward
pass while memory-bandwidth-bound at low batch size. More available
bandwidth on H200 doesn't remove that halving, but it can shrink its
*share* of total TTFT if other fixed overheads (kernel launch, Python/
async overhead in the harness's own request path) don't shrink at the
same rate -- so the low-load FP8 edge may be smaller in absolute ms, or
harder to distinguish from noise, than the already-marginal H100 number
was.

**Superseded -- the paragraph above assumed weight-only FP8, matching
the H100 arm, and reasoned entirely in terms of memory-bandwidth relief.
PR #24 found that's wrong for what's actually running here:** vLLM
0.19.1's plain `fp8` is W8A8 with dynamic activation scaling -- weights
*and* activations both quantized, scale recomputed every forward pass
(see "FP8 on vLLM 0.19.1" below). That's a different mechanism, not the
same mechanism in a roomier memory budget, and it changes the
prediction rather than just adjusting its magnitude.

**Predicted (this supersedes the FP8-specific paragraph above, not just
adjusts it):** dynamic activation quantization is a per-forward-pass
compute cost -- quantize, compute a scale, dequantize -- that doesn't
shrink with more VRAM, because it was never a memory-capacity problem to
begin with. More HBM and more bandwidth do nothing to relieve a cost
that's compute, not memory. **The H200 FP8 arm may be slower than the
H100 FP8 arm's numbers at every load level tested, including low load,
where the H100's weight-only arm had its one small real win (8.23 vs
8.39ms).** If so, that isn't H200 underperforming H100 -- it's two
different interventions sharing an arm label, one of which does
strictly more work per token than the other.
**What would falsify this:** the H200 FP8 arm's low-load TTFT p50
coming in at or below the H100 FP8 arm's 8.23ms would be the surprise --
it would mean the dynamic-activation-quantization overhead is smaller
in practice than the mechanism argument above predicts, and that's
worth profiling rather than waving off as "H200 is just faster."
Separately: if the numbers come back and don't show a compute-bound-
shaped penalty at all (e.g. the H200 FP8 arm tracks fp16 closely at
every load point), the first thing to check is whether the online W8A8
path is actually the one running at request time -- re-verify with the
same live-config-read method PR #24 used, not assumed to still hold.

**Predicted, and the reason this isn't just "H200 is faster so multiply
by a constant":** concurrency ~=1/8/32 is derived per-arm from that
arm's own H200 service time via `scripts/calibrate.py` (Little's Law),
same procedure as the H100 run. If H200's extra bandwidth changes
low-load service time by a different factor than it changes the
compute-bound regime's onset, "concurrency ~= 32" on H200 is not
guaranteed to land at the same point on the memory-bound-to-compute-
bound curve that "concurrency ~= 32" landed at on H100 -- the label is
portable, the physical regime it names might not be. If quantization
effects at c~=32 look different in kind (not just degree) from the H100
result, this is the first place to look before concluding the mechanism
changed.

**If the data contradicts any of this** -- AWQ or FP8 showing a larger
benefit on H200 than H100 at the same nominal concurrency -- that's a
more interesting result than confirmation would be, and gets written up
as such rather than smoothed over, same standard as the H100 predictions
above.

## FP8 on vLLM 0.19.1: not the same recipe as the H100's fp8_per_tensor

**Predicted going in:** that it might differ -- the H100 FP8-flag entry
above already flagged this as something to re-check on any vLLM version
change, not assume.
**Found, by reading the resolved config, not the docs or a comment**
(same method as the original H100 investigation): constructed the exact
config `vllm serve --quantization fp8` produces for
`Qwen/Qwen2.5-1.5B-Instruct` via `EngineArgs(...).create_engine_config()`
and inspected the resulting `Fp8Config` instance directly --
`activation_scheme: dynamic`, `is_checkpoint_fp8_serialized: False`.
Confirmed against the source
(`vllm/model_executor/layers/quantization/fp8.py`): the online path
(`Fp8OnlineLinearMethod`, used here since this is an unquantized bf16
checkpoint being quantized on the fly, same as the H100 arm) sets
`self.act_q_static = self.quant_config.activation_scheme == "static"`,
and `activation_scheme` defaults to `"dynamic"` -- there is no CLI
override that changes this for the bare `fp8` shorthand.
**This means: activations are quantized too, dynamically -- this is
W8A8, not weight-only.** The H100's `fp8_per_tensor` was confirmed
weight-only (`activation=None` in its own resolved
`QuantizationConfigArgs`, see the H100 FP8-flag entry above): weights
move to fp8, activations stay in bf16. vLLM 0.19.1's plain `fp8` instead
quantizes both weights *and* activations to `float8_e4m3fn`, with the
activation scale computed dynamically per forward pass rather than a
static value. **These are two different quantization recipes with the
same arm label.** Any H200 FP8 number is not measuring the same
intervention as the H100 FP8 number, on top of already not being
comparable for hardware/vLLM-version reasons (see the hardware-change
and environment-rebuild entries above) -- this is a third, independent
reason, specific to what the FP8 arm even *is* on this pod.
**Why it matters:** W8A8 dynamic and weight-only static have different
expected performance shapes. Weight-only's win case (argued in the
original H100 prediction) is memory-bandwidth-bound low-load decode,
where only smaller weights matter and activations passing through
unquantized costs nothing. W8A8 adds a real compute cost the H100 arm
never paid -- quantizing activations on the fly, every forward pass, at
every load level -- so the H200 FP8 arm's low-load number shouldn't be
expected to reproduce the H100's small win, and its high-load number
isn't testing the same "does weight-only pay off when compute-bound"
question the H100 result answered. Any README claim about H200 FP8 has
to say W8A8-dynamic, not carry over "weight-only" from the H100 section.
**Also recorded, since the H100 run never captured it:** AWQ's kernel
selection on this environment -- `The model is convertible to
awq_marlin during runtime. Using awq_marlin kernel.`, then `Using
MacheteLinearKernel for AWQMarlinLinearMethod`. No equivalent H100-side
log capture exists to compare against (another gap issue #19's server-
log-capture fix will close going forward), so this is a baseline for
future comparison, not a diff against something already known.
**How to apply:** when the H200 sweep actually runs and this FP8 arm's
numbers get written up, the README/decisions.md scope note has to say
"weight-and-activation FP8, dynamic scale (W8A8)" for the H200 arm,
distinct from "weight-only, static scale" for the H100 arm -- not reuse
the H100 arm's FP8 description. If a future vLLM version on either
environment changes this again, re-verify the same way rather than
assuming either recipe carries forward.
## The H200 run is a fresh baseline, not a continuation

**Found, once the redeploy attempt confirmed the driver:** a second H200
pod was requested specifically to test whether the CUDA-13/driver
mismatch (see "H200 environment rebuild" above) was a property of the
first pod or of RunPod's US-NC-1 H200 allocation generally. The
redeployed pod reports the same driver (`570.124.06`), and the same
container hostname (`3a4e9ae6f924`) as the original -- RunPod appears to
have handed back the same physical host, not different hardware. vLLM
0.19.1 remains the ceiling; 0.26.0 is not reachable on this allocation.
**Chose:** stop trying to eliminate the software variable and proceed on
0.19.1, per the standing instruction that a second 570-series result
ends the attempt rather than prompting a third.
**Why it matters, stated as its own entry rather than left implicit
across three separate ones:** the H200 run does not differ from the
H100 run in one way (hardware). It differs in **four**, independently:
- **GPU** -- H100 80GB HBM3 vs H200 SXM 141GB (see "Hardware change").
- **vLLM version** -- 0.26.0 vs 0.19.1, forced by the driver, not chosen
  (see "H200 environment rebuild").
- **Attention backend** -- flashinfer 0.6.14 vs FlashAttention v3, a
  consequence of the version difference, not set directly (same entry).
- **FP8 semantics** -- weight-only static-scale vs W8A8 dynamic-scale, a
  consequence of the vLLM version difference reaching all the way into
  what the FP8 arm's flag *means* (see "FP8 on vLLM 0.19.1").
Any one of these alone would be enough to block a direct comparison.
Together, "the H200 run" is not the H100 run measured again on newer
hardware -- it's a different measurement that happens to share a
workload design (`harness.py`, the arm/load/barge-in/prefix-caching
matrix) with the H100 run, not its results.
**Chose:** treat the H200 sweep as a fresh baseline. The H100 results
stay in this repo, unchanged and undeleted, clearly labeled as a prior
run with their own environment block -- not superseded, not silently
replaced. **No table, plot, or claim in this repo mixes H100 and H200
numbers.** Where a comparison between them is interesting (e.g. "is
FP8 still a net win on newer hardware"), it gets stated as a comparison
between two different, named environments, with the four differences
above listed alongside it -- never as a single number implying nothing
changed but the GPU.
**Rejected:** treating the H200 run as simply "the same sweep, re-run."
That framing would silently launder four real differences into what
looks like a hardware-only comparison, which is exactly the kind of
thing this repo's own stated rule (predict before measuring, provenance
on every result) exists to prevent.
**How to apply:** a repo that documents a hardware/environment change
and keeps both result sets, clearly separated, reads as rigor. A repo
that quietly drops the old numbers and starts over reads as something
went wrong and got hidden. The old results are not a mistake to erase;
they're the H100 baseline, still true for the hardware that produced
them.

## FP8's ~4x token-length gap is not a quantization effect -- it's a bug

The H200 calibration checkpoint (`results/h200/calibration_fp8*.json`)
showed FP8 averaging ~78 tokens/response against fp16's ~20-24 -- large
enough, and the two distributions separated cleanly enough (FP8: 74-80
tokens, 98%+ within 2 of the `max_tokens=80` cap; fp16: 10-80 tokens,
median 20, 0.2% at cap), that treating it as "W8A8 costs more compute
per token" without checking further would have been asserting a story
the aggregate timing couldn't actually distinguish from a bug. Checked
directly, four ways, before writing anything into a table:

**1. `finish_reason`.** Not previously captured (issue #8, now fixed --
see that PR). Fixed, then measured directly against a live server, both
arms, same fixed prompt/turn set, no aggregation: **fp16: 221 `stop` /
1 `length` (99.5% natural EOS). FP8: 0 `stop` / 76 `length` (100% hitting
the cap, zero natural stops in the sample).** Not a skew, not a
tendency -- FP8 never once reached EOS on its own.

**2. Chat template.** Both arms load the identical model repo
(`Qwen/Qwen2.5-1.5B-Instruct`; FP8 quantizes it on the fly, doesn't
swap the checkpoint), so the tokenizer and template are the same files
by construction -- confirmed rather than assumed: both startup logs
show the identical line, `Detected the chat template content format to
be 'string'`. No divergence to find.

**3. EOS / sampling config resolution.** Both startup logs show the
identical `generation_config.json` override
(`{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p':
0.8}`), same warning, same values, same source file (same model repo).
Nothing arm-specific in how stop conditions get resolved.

**4. Output text.** Sent the same system prompt + user turn to both
arms directly (`/v1/chat/completions`, non-streaming, so the full text
is visible at once), repeated across two different prompts and two
seeds. fp16, `seed=0`: *"It's okay to spend time on thoughts that
matter to you. What is the thought or idea you've been pondering?
Sometimes talking through what's on your mind can help clear things
up."* -- coherent, on-topic, `finish_reason: stop`, 40 tokens. FP8, same
prompt, `seed=0`, `finish_reason: length`, 80 tokens: *"袅 解.Resolve";}
yabǃ qualidade感じるolving (...ñas andaLab yab standardized zwe...` --
mixed-script token soup, not language. A second prompt at `max_tokens=8`
(`seed=1`) confirms this isn't a late-sequence drift: *" Peripheral
resolveolving";} yab感じる獬 qualidade"* -- garbled from the first
token.

**Conclusion: none of the four are ambiguous, and none point at
configuration.** Chat template and EOS/sampling resolution are
confirmed byte-identical between arms -- ruled out. `finish_reason` and
the raw text together rule out "real quantization effect" too: genuine
W8A8 quantization noise degrades coherence at the margins, it doesn't
produce multi-script token soup from the first generated token with
100% cap-hitting across every sample. **This is vLLM 0.19.1's online
FP8 (`Fp8OnlineLinearMethod`, W8A8 dynamic activation scaling --
see "FP8 on vLLM 0.19.1" above) producing corrupted output, not a
subtle latency or quality cost of quantization.**
**Candidate mechanism, a hypothesis, not confirmed by profiling:** the
H100 arm's online FP8 (`fp8_per_tensor`, weight-only, static scale) was
also computed on the fly from an unquantized checkpoint and produced
normal output -- so "online quantization is inherently unreliable"
doesn't fit; the H100 case rules that out. The variable that's actually
new here is *dynamic activation* quantization -- computing an
activation scale fresh every forward pass, not just quantizing weights
once at load time. That's the most likely place a real bug lives,
but confirming it would need kernel-level inspection, out of scope here.
**What this means for the sweep:** the FP8 arm, as configured
(`--quantization fp8` on vLLM 0.19.1), does not produce usable output.
Any latency number from it describes how fast this server generates
corrupted text, not a valid FP8 comparison point -- weight-only vs W8A8
was never going to be comparable to the H100 arm (see "FP8 on vLLM
0.19.1"), but this is a different, more basic problem: **the current
FP8 configuration is broken, not just differently-scoped.** Recommend
excluding it from the H200 sweep matrix rather than measuring it,
pending either an upstream vLLM fix/version bump or finding a working
FP8 configuration on this environment -- neither attempted here.
**Also found, stated plainly since it affects every number already
committed under `results/h200/calibration_fp8*.json`:** FP8's derived
rates (5.4647 / 43.7178 / 174.8711 req/s) came from Little's Law reading
the corrupted-output service time (183ms mean, vs fp16's 67ms) as if it
were a real measurement. **Those rates are not valid and must not be
used to drive a load-test run, or presented as FP8's arrival rate at
equivalent concurrency to the other arms.** The calibration files stay
committed (the measurement itself -- what the server actually did under
that config -- is accurately recorded; the *interpretation* of it as
"FP8's service time" is what's wrong), with this entry as the reason not
to build anything on them as-is.
**How to apply:** if a working FP8 configuration is found later on this
environment, it needs its own calibration pass and its own
`finish_reason` check before being trusted -- this entry's fix
(capturing `finish_reason`) stays in the harness permanently specifically
so this class of problem surfaces on the first calibration checkpoint
next time, not after a full sweep's worth of GPU time.

## H200 sweep results (fp16 + AWQ, FP8 excluded per issue #29)

Ran with the calibration checkpoint's own derived rates (not recalibrated
-- that measurement was already valid) and the four sweep-v2 fixes.
Full matrix: `results/h200/summary.md`. One anomaly caught and fixed
before anything else: `fp16_pcoff_open_c8_bargein0.0` initially showed
TTFT p99 = 5.3s (581 of 2178 requests over 1s) -- a linear queueing
drain from ~5.1s mean TTFT at t=0 down to steady-state ~16ms by t=7s,
then flat for the remaining 13s. Checked whether this was systemic
(every other cell's own load-level transition, including AWQ's and
fp16-prefix-on's own c1->c8 jumps) -- none of the other 34 result files
show anything like it (max TTFT elsewhere: tens of ms, this one: 5.5s).
Isolated to this one run. Re-ran it alone (same config, fresh server):
clean (p50 14.9ms, p99 41.4ms, in line with every other c8 cell). Kept
the re-run, not the original -- one-off host/GPU hiccup, not a load
level that's actually unstable at c8 (every other c8 cell, before and
after, is clean).

**Does AWQ's low-load penalty persist? Yes. Does the predicted
high-load benefit stay smaller than H100's? No -- it's larger, and in
the opposite shape from what H100 showed.**
- c~=1: fp16 11.29+/-0.18ms vs AWQ 12.93+/-0.18ms -- AWQ +14.5% slower,
  clearly outside the noise bands (H100: 8.39 vs 10.47ms, +24.8%). Same
  direction, smaller relative gap, but still unambiguous -- persists.
- c~=32: fp16 45.59+/-2.81ms vs **AWQ 35.76+/-0.73ms -- AWQ ~21.6%
  *faster***, both repeat-validated (5 seeds each), well outside
  combined noise. H100 showed no measurable difference at c~=32
  (37.79 vs 37.31ms, inside noise). **This contradicts the pre-registered
  prediction** ("AWQ's fixed dequant overhead should cost roughly the
  same absolute time on H200... AWQ's relative low-load penalty should
  hold or widen, not shrink" -- silent on high load beyond "smaller
  benefit than H100's ~0"). H200's AWQ arm doesn't just fail to shrink
  its high-load gap, it flips from "no difference" to "clearly ahead."
  **Candidate mechanism, a hypothesis, not confirmed by profiling:** the
  prediction's dequant-is-a-fixed-compute-cost argument was about low
  load. At c~=32, if the workload is memory-bandwidth-bound rather than
  compute-bound, AWQ's int4 weights (4x fewer bytes than fp16's bf16)
  moving less data per forward pass could be a real bandwidth win --
  the same mechanism the *H100 FP8* prediction argued for weight-only
  quantization, just now showing up for AWQ instead, on hardware with
  more bandwidth headroom to exploit it. Whether H200's specific
  `awq_marlin`/`MacheteLinearKernel` path (recorded in
  `results/h200/server_log_awq.txt`) is simply better-optimized than
  whatever ran on H100 is a separate, untested alternative explanation.
  This is a real, repeat-validated result either way -- the mechanism is
  open, the direction and magnitude aren't.

**Does the c~=1 prefix-caching reversal reproduce? Weakly, within
noise -- not the same clean effect as H100's.**
H100: +12.4% p50 / +4.4% p99 at c~=1 (caching on measurably worse).
H200: +0.5% p50 / -0.6% p99 at c~=1 -- both inside the off-arm's own
repeat noise band (+/-0.18ms on a 0.06ms swing). Present in direction at
p50, absent at p99, neither clearly outside noise. The rest of the
pattern holds cleanly: c~=8 -3.2%/-51.1%, c~=32 -32.8%/-39.8% (vs H100's
-6.1%/-27.6% and -27.8%/-37.8% -- same shape, comparable-to-larger
magnitude). **The lead finding (caching matters most at high
concurrency) reproduces solidly; the specific low-load reversal doesn't
reproduce as a distinct, trustworthy effect on this hardware/vLLM
version -- report the c~=32 number with confidence, don't lean on the
c~=1 reversal as if H200 confirmed it.**

**Is barge-in landing mid-generation now? Yes, confirmed directly --
this was the entire point of the fix.** H100: `aborted_total` was 0 at
every c~=1 and c~=8 `bargein0.25` run. H200: abort rate is 25-27% at
**every** concurrency (c1/c8/c32), matching the sampled 25% fraction
almost exactly -- the scaled window (25-75% of each arm's calibrated
service time) reliably fires now. Checked further, not just "it fires":
`abort_before_first_token` is 0.0-0.9% at c1/c8 (aborts land after TTFT,
during decode -- genuinely mid-generation) but 11.6-26.5% at c32 --
exactly the documented limitation (`scripts/sweep.sh`'s own comment):
the window is sized off *unqueued* service time, so at c~=32, where
queueing measurably inflates actual wait time, a real fraction of
"25-75% of unqueued service time" lands during the queued TTFT phase
rather than post-TTFT decode. Predicted in the code comment before this
run; confirmed by the data now, not asserted.

**`finish_reason` distribution, now captured (#28): clean across every
fp16/AWQ cell.** Overwhelmingly `stop` (natural EOS); a small `length`
tail (~0.1-0.3% of completed requests, expected variance -- some
responses are legitimately verbose); `null` exactly where aborts
happened (matches `aborted_total` 1:1 in every file checked -- aborted
requests never receive a finish_reason chunk). No arm, no cell, shows
anything resembling FP8's 100%-`length` pattern. Nothing to investigate
further -- this is what a healthy arm's distribution looks like, the
contrast that makes FP8's finding legible as a bug rather than noise.

## KV-cache arithmetic check: passes (issue #19 closed)

First time this check has had anything to verify against --
`scripts/sweep.sh`'s server-log capture (`results/h200/server_log_*.txt`)
made it possible. Checked both fp16 and AWQ's startup logs directly.
fp16: `Available KV cache memory: 120.53 GiB`, `GPU KV cache size:
4,513,888 tokens` -> vLLM's own implied bytes/token = (120.53 * 2^30) /
4,513,888 = **28,671.09**. `scripts/kv_cache_check.py`'s formula (28
layers x 2 KV heads x 128 head_dim x 2 (K/V) x 2 bytes, `kv_cache_dtype=
auto` resolving to the model's own 2-byte dtype) = **28,672**. Difference:
-0.003%, fully explained by vLLM's log rounding "GiB" to 2 decimal
places before printing -- working backward from a rounded display value
can't recover more precision than that. AWQ: `Available KV cache memory:
122.29 GiB`, `4,579,824 tokens` -> implied 28,670.95 vs the same 28,672
expected (AWQ quantizes weights, not the KV cache -- `dtype=torch.float16`
here vs fp16 arm's `bfloat16`, but both are 2-byte formats, so the
byte-per-token arithmetic is unaffected either way) -- same match,
-0.004%.
**They agree.** The arithmetic in `scripts/kv_cache_check.py` was
correct when it was first written with nothing to check it against;
now there's something, and it holds.

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

## Perplexity measurement design (Phase 4)

**Chose:** a fixed slice of the wikitext-2-raw-v1 test set (8,192 tokens,
tokenized once with `Qwen/Qwen2.5-1.5B-Instruct`'s own tokenizer via
`scripts/build_perplexity_slice.py`, committed as both the raw token ids
(`data/wikitext2_test_slice_token_ids.json`, what actually gets sent to the
server) and the decoded text (`data/wikitext2_test_slice.txt`, so a reader
can see what's being evaluated without running anything) -- and
forced-decoding against a live vLLM server: the slice's own token ids sent
as the prompt with `max_tokens=0, echo=True, prompt_logprobs=0`. The server
never samples anything; for each token after the first it reports the
log-probability it assigned to that exact real token given everything
before it. NLL = -mean(those logprobs); PPL = exp(NLL).
**Rejected:** loading the model offline (outside vLLM) via `transformers`
or vLLM's `LLM` class and computing perplexity directly. That would measure
a different code path than the one actually serving requests in every
other table in this repo -- the whole point of this repo's provenance
discipline is that the server that ran the workload is the thing being
measured, not a proxy for it. `prompt_logprobs` gets the same number from
the same serving stack, for free, no extra dependency.
**Confirmed against a live server before writing `scripts/perplexity.py`,
not assumed from the vLLM source alone:** sent `{"prompt": [token ids],
"max_tokens": 0, "echo": true, "prompt_logprobs": 0}` to a running fp16
server's `/completions` endpoint. Response `choices[0].prompt_logprobs` is
a list, one entry per input token: position 0 is `null` (no preceding
context to condition the first token on -- excluded from the mean, not
zero-filled), positions 1..n-1 are each `{"<token_id>": {"logprob": ...,
"rank": ..., "decoded_token": ...}}` -- exactly the actual input token's
own logprob, no sampling, no generated text. `scripts/perplexity.py`
asserts this shape on every call (right token id present at every
position, no unexpected nulls) rather than trusting it silently.
**Why the token count (8,192), not the full wikitext-2 test set (~300k
tokens) or a much shorter one:** a single forced-decoding call is one
prefill pass, not autoregressive generation -- cost scales with tokens
processed once, not with a sequential per-token decode loop, so even the
full test set would run in one request. 8,192 was chosen for a stable
per-arm estimate (over a hundred articles' worth of tokens, not a
handful of sentences) while keeping repeat runs (needed for the noise
band, see below) cheap. Not stride-windowed: the 32,768 context limit
comfortably covers 8,192 tokens in one pass, so there's no need for the
overlapping-window scheme papers use to handle context limits shorter
than the eval set -- every token in the slice is scored with its full
preceding context, the unbiased case, not a proxy for it.
**Why repeats, when nothing here samples anything:** identical token ids,
identical weights, identical server, zero sampling -- so a naive
expectation is that PPL should come back bit-identical every time.
Whether it actually does is an empirical question about GPU kernel
determinism (parallel reduction order, batching-dependent kernel
selection), not something to assume either way. `scripts/perplexity.py`
runs `--repeats` (default 5, matching this repo's other noise-band
passes) identical calls and reports mean/stdev across them -- if the
stdev comes back exactly zero, that is itself a finding, not evidence
the check was pointless.
**How to apply, corrected after actually running it:** the stdev came
back exactly zero (see "Phase 4 perplexity, single slice repeated,"
below) -- a real methodology finding (this serving path is
bit-reproducible under these conditions), but a band of zero supplies no
error bar an AWQ-vs-fp16 delta can be checked against. Re-running one
fixed slice more times was never going to produce one; a deterministic
computation has no variance to average out. The actual noise band this
repo's own standard (`docs/decisions.md`'s "Corrected results" entry --
delta vs. run-to-run spread, before calling anything a real difference)
requires came from a different axis instead: scoring several distinct
text slices once each and taking the spread across *them*. See "Phase 4
perplexity, cross-slice," below.
**Limitation, stated plainly:** wikitext perplexity measures next-token
prediction on encyclopedic prose. This project's actual workload
(`harness.py`'s `USER_TURNS`) is short-form multi-turn conversational
dialogue -- a different distribution, different sequence lengths, no
back-and-forth turn structure. A wikitext PPL delta says something
directionally real about representational quality lost to quantization,
not a validated statement about this avatar workload's actual response
quality. Any writeup pairing this number with the latency trade has to
say so, not imply wikitext PPL is a stand-in for conversational quality.

## Predictions, stated before measuring (Phase 4: perplexity)

Qwen2.5-1.5B-Instruct AWQ int4, group-wise scaling (the standard AWQ
recipe -- weights quantized to 4 bits with a per-group scale fit to
minimize activation-weighted quantization error, not a plain round-to-
nearest). Group-wise scaling bounds representational error per group,
which keeps a well-calibrated 4-bit method's quality cost small, but 4
bits is still a real information loss versus fp16 -- some measurable
degradation is the expected default, not "no difference."

**Predicted:** AWQ's perplexity on the fixed wikitext-2 slice comes back
higher than fp16's (worse), by a small relative margin -- order of a few
percent relative to fp16's PPL, not zero and not large. fp16 itself is
expected to have some nonzero run-to-run stdev from kernel
nondeterminism (see the entry above); AWQ's mean is predicted to sit
outside that combined noise band, not just nominally higher.

**What would be the surprise, stated in both directions before
measuring, same as every other prediction in this file:**
- **No measurable difference** (AWQ's mean PPL inside the combined
  fp16/AWQ noise band) would be a real result worth reporting as such --
  it would mean 4-bit AWQ costs nothing detectable on this metric for
  this model size, which would make the latency trade at c~=32 (AWQ
  21.6% faster, repeat-validated) look close to a free win rather than a
  real trade. Surprising relative to the "4 bits is real information
  loss" default, not implausible -- AWQ's whole design goal is exactly
  this outcome.
- **A large relative increase (order tens of percent or more)** would be
  the other surprise -- that would look less like a normal quantization
  cost and more like something closer to broken, the same category of
  finding issue #29 turned up for FP8's corrupted output. If this
  happens, the first move is the same four-way check that entry used:
  read a few actual decoded completions from the AWQ arm before trusting
  the number, not just the aggregate PPL.
- The genuinely expected outcome -- small but real, outside the noise
  band -- is the least interesting of the three to write up on its own,
  but it's what pairs with the latency finding to make the actual point:
  AWQ is not a free lunch at c~=32, it trades a small, real quality cost
  for the 21.6% latency win, and it still pays the 14.5% low-load latency
  penalty on top of that same quality cost -- worse on both axes at low
  load, a real trade at high load.

FP8 is excluded from this measurement, same as every other H200 table --
issue #29's corrupted output means there is no coherent text to measure
the perplexity of; a PPL number from it would describe confidence in
token soup, not quality.

## Phase 4 perplexity, single slice repeated: a determinism check, not a noise band

`results/h200/perplexity_fp16.json`, `results/h200/perplexity_awq.json` --
5 repeats each, fresh server per arm, one fixed 8192-token wikitext-2
slice, forced-decoding (`scripts/perplexity.py`).

**Every one of the 5 repeats, for both arms, returned bit-identical NLL
and perplexity (stdev 0.0).** That result is a property of the
measurement, not a property worth calling a finding about the system on
its own: forced-decoding a fixed slice against fixed weights involves no
sampling anywhere -- same tokens in, same logits out, every time, by
construction. Repeating an experiment that has no source of randomness
can't produce variance; getting exactly zero back is confirmation the
serving path is bit-reproducible under these specific conditions (one
unbatched sequence, idle server, no concurrent requests perturbing kernel
batching/reduction order), not evidence about the size of an effect.
**This run supplies no noise band.** fp16: PPL 9.7118. AWQ: PPL 10.5145 --
an 8.26% relative gap on this one slice, but "outside a noise band of
zero" isn't a meaningful claim; a band of zero can't be beaten by
anything. Kept as a methodology note (the serving path's own
determinism, worth knowing on its own terms) and as the reason the actual
uncertainty estimate had to come from somewhere else -- see below, not
from repeating this measurement more.

## Phase 4 perplexity, cross-slice: the real noise band, and the actual finding

**At low load, AWQ is worse on both axes it will ever be judged on --
slower (+14.5%, repeat-validated, `docs/decisions.md`/"H200 sweep
results") and lower-quality (+8.02% perplexity, cross-slice, below).
There is no concurrency in this sweep's tested range where AWQ is the
right choice at c~=1.** That's the headline; the c~=32 trade below is the
more interesting mechanism, but this is the more decision-relevant fact.

**Method:** `scripts/build_perplexity_slices.py` cuts 8 non-overlapping
8192-token slices from wikitext-2's test set (same tokenizer, same source
file as the single-slice version -- different token ranges, not different
sampling). `scripts/perplexity_multislice.py` forced-decodes each slice
once per arm (no repeats -- the entry above already established that
repeating one slice adds nothing) and reports mean/sd of perplexity
*across slices*. That cross-slice spread is the uncertainty that actually
applies to a claim like "AWQ costs 8% perplexity" -- how much the number
moves when the underlying text changes, not whether re-running the exact
same forward pass changes anything. `results/h200/
perplexity_multislice_fp16.json`, `perplexity_multislice_awq.json`.

**Per-slice results (PPL):**

| Slice | fp16 | AWQ | AWQ relative to fp16 |
|---|---|---|---|
| 0 | 9.7118 | 10.5145 | +8.27% |
| 1 | 5.9663 | 6.4418 | +7.97% |
| 2 | 8.7479 | 9.4362 | +7.87% |
| 3 | 8.7651 | 9.4873 | +8.24% |
| 4 | 8.7061 | 9.3278 | +7.14% |
| 5 | 9.1789 | 9.9239 | +8.12% |
| 6 | 11.3841 | 12.2808 | +7.88% |
| 7 | 7.8761 | 8.5625 | +8.71% |

**Cross-slice: fp16 8.7921 +/- 1.5375, AWQ 9.4969 +/- 1.6565.** Raw
perplexity itself swings hard across slices -- 5.97 to 11.38 for fp16,
nearly a factor of 2 -- entirely from text difficulty (some wikitext
articles are more predictable than others); this is expected and not
itself a finding about either arm.

**The gap holds consistently, not just on average -- this is the actual
result, not the mean-of-means above.** AWQ is worse than fp16 on all 8 of
8 slices, no exceptions, and the *relative* penalty is tight even where
the raw numbers swing wide: 7.14% to 8.71%, mean 8.02%, sd 0.45 percentage
points across slices. The per-slice relative comparison controls for
each slice's own difficulty and isolates the quantization cost cleanly --
which is why it's a better-supported number than either arm's raw
cross-slice mean/sd above, and why "the claim is solid for the right
reason" (the ~8% cost shows up regardless of what text it's measured on,
not just on the one slice originally tried).

**Against the pre-registered prediction:** predicted "a few percent,"
landed at 8.02% (cross-slice mean of the per-slice relative deltas,
superseding the single-slice 8.26% above as the number to actually cite).
Direction right, order of magnitude right, on the high side of "a few" --
same read as before, now on firmer footing.

**Paired with the latency finding this was measured to complete
(docs/decisions.md, "H200 sweep results," and the README's AWQ section):**
at c~=1, AWQ is worse on both axes -- 8.02% worse perplexity, 14.5% slower
-- never the right choice at low load in this sweep's range. At c~=32,
AWQ trades that same ~8% perplexity cost for a 21.6% TTFT p50 win -- a
real, quantifiable trade, both sides repeat-validated. The quality cost
doesn't change with load (it's a property of the weights, not the batch);
whether it buys anything back in latency depends entirely on where you
are on the load curve.

**Limitation, stated plainly:** this is wikitext-2 next-token perplexity,
encyclopedic prose, not this project's actual multi-turn conversational
workload. An 8.02% PPL increase says AWQ's weights carry real, measurable
representational error relative to fp16 -- it does not say how a human
rating this avatar's actual responses for coherence or helpfulness would
perceive that error, which could be smaller (short conversational replies
may not expose the kind of long-range prediction fp16 does better) or
larger (a single bad word choice is more noticeable in a four-sentence
reply than averaged into an 8192-token encyclopedia excerpt) than this
number implies either way. Reported as what it is -- a real, repeatable
signal that AWQ costs something on a standard proxy metric, now with an
actual cross-sample error bar behind it -- not as a validated statement
about conversational quality.

## Phase 6 model choice: Stable Diffusion 1.5, not a video model

**Chose:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (a maintained
mirror of `runwayml/stable-diffusion-v1-5`, identical weights -- same
resolved commit SHA on both, confirmed via `HfApi.model_info` before
picking one), fp16, DPM-Solver++ (`DPMSolverMultistepScheduler`),
512x512 -- SD1.5's native trained resolution, not a chosen-for-the-story
number. Runs entirely through `diffusers` -- no vLLM, no server, no HTTP;
`torch.cuda.synchronize()` around each stage in a plain Python process.
**Rejected:** a genuine video diffusion model (AnimateDiff, SVD,
CogVideoX). This project's actual subject is *frame-by-frame* real-time
generation -- the 40ms budget in the prompt is a per-frame number, which
is how a real streaming avatar system would have to work (one frame
conditioned on the latest driving signal, not a whole clip generated
ahead of time) -- so a single-image text-to-image UNet model is the
right proxy shape, not a compromise. A real video model adds temporal
attention/frame-conditioning machinery this project has no way to
validate is being used realistically, for no benefit to the actual
question being asked (per-frame stage cost).
**Rejected:** SDXL. Heavier UNet, native 1024x1024 -- both push per-step
cost up, which sounded like it would produce a more "production-scale"
number, but a pilot run (not the committed measurement, see below)
showed SD1.5 already lands within a small factor of the 40ms budget at
step counts as low as 1 -- the tension the ceiling question is about
(does anything fit, and by how much) is visible at SD1.5's native
resolution without needing to reach for a heavier model to make the
constraint feel real.
**Rejected:** a turbo/distilled variant (SDXL-Turbo, SD-Turbo, LCM).
These are trained specifically to be good at 1-4 steps -- using one
would answer "how fast can a model *already built for* the low-step
regime go," not "what happens when you point a standard, undistilled
diffusion model at a real-time budget," which is the more informative
question for a stage-cost breakdown: a turbo model's designers already
made the VAE-decode-vs-step-count tradeoff invisible by construction
(few steps is the intended operating point, so nothing here would be
surprising). A standard model exposes the actual tension.
**Scheduler:** DPM-Solver++ specifically because it's designed to reach
reasonable quality in fewer steps than the DDPM-style schedulers SD1.5
shipped with originally -- the natural choice for a "how few steps can
this get away with" question, not an arbitrary pick.
**This is a proxy for real-time avatar video generation, not the thing
itself,** stated here and restated in the README when this phase's
findings land there: no lip-sync, no audio conditioning, no temporal
consistency across frames, no actual avatar-specific model -- a
standard text-to-image diffusion model's per-frame cost structure,
measured under the same 40ms/100ms constraint a real system would face.
What generalizes: the stage-cost shape (fixed conditioning + VAE cost,
step count as the one lever that scales). What doesn't: absolute
milliseconds for any actual avatar-specific architecture, which this
project has not measured and does not claim to.

## Phase 6: pre-registered hypothesis (stated before the committed measurement)

A pilot run (`torch.cuda.synchronize()`-timed, discarded, not the
committed measurement below -- used only to confirm the timing
methodology itself works before committing to it) surfaced something
that has to be handled before any real number gets reported: the
**first** call to every stage -- text encoder, first UNet step, VAE
decode -- is dominated by CUDA/cuDNN kernel warmup, not real cost
(conditioning: 114.73ms unwarmed vs 6.2-6.7ms warmed; first denoising
step: 298.81ms unwarmed vs ~18ms steady-state). Every run in the
committed measurement below discards one full warmup generation before
any timed run, the same discipline `scripts/calibrate.py` and
`scripts/sweep.sh` already use for the LLM sweep (`--warmup`), applied
here for the same reason: an unwarmed first call isn't a real number,
it's a one-time cost that amortizes to nothing over a real streaming
session.

**Predicted, from the pilot's warmed numbers and from the architecture,
not just the pilot in isolation:** VAE decode is a single fixed-cost
forward pass through a deep convolutional network, independent of how
many denoising steps ran before it -- structurally, it's the same shape
of cost as *one* denoising step (also a single forward pass through a
comparably-sized network), just through a different network. If that
holds, VAE decode should be comparable in magnitude to a single
denoising step, not a rounding error next to it -- and at low step
counts (where total time is conditioning + a small N x step_cost + a
fixed VAE cost), the fixed VAE cost should make up a large,
non-shrinking share of total frame time. The pilot's warmed numbers are
consistent with this: ~21ms VAE decode against an ~18-23ms single
step -- roughly the same order of magnitude, not one dwarfing the other.

**What would falsify this:** VAE decode taking a small fraction (order
10-20% or less) of a single denoising step's time, at any step count --
that would mean step count is the only real lever, and VAE decode is a
correctly-ignorable fixed cost, contradicting the prediction. The
committed measurement (`scripts/diffusion_bench.py`, repeats +
`torch.cuda.synchronize()` around every stage, `results/h200/diffusion/`)
is what actually decides this, not the pilot -- report below.

## Phase 6 results: the fixed cost, not the ceiling, is the finding

512x512, fp16, DPM-Solver++, 5 repeats per cell, warmup discarded
(`results/h200/diffusion/steps{1,2,3,4,5,8,12,20}.json`):

| Steps | cond (ms) | step mean (ms) | VAE decode (ms) | total (ms) | total sd |
|---|---|---|---|---|---|
| 1 | 6.61 | 18.74 | 20.98 | 46.33 | 0.82 |
| 2 | 6.62 | 18.65 | 20.96 | 64.88 | 0.42 |
| 3 | 6.52 | 18.98 | 20.97 | 84.42 | 2.65 |
| 4 | 6.48 | 18.35 | 21.00 | 100.89 | 0.61 |
| 5 | 6.99 | 20.88 | 21.03 | 132.42 | 1.59 |
| 8 | 6.47 | 18.27 | 21.01 | 173.63 | 0.33 |
| 12 | 6.38 | 17.87 | 21.00 | 241.76 | 0.36 |
| 20 | 6.56 | 18.17 | 21.02 | 390.99 | 1.57 |

**The finding is not "how many steps fit" -- it's that most of the
budget is gone before the first step runs.** Mean conditioning (6.58ms)
+ mean VAE decode (20.996ms) across all eight cells = **27.57ms fixed
cost, 68.9% of the 40ms/frame real-time budget, that no step-count
choice touches.** Step count is the lever the field's own optimization
literature focuses on (this is exactly why DeepCache/TeaCache-style
methods target the denoising loop specifically) -- at this resolution,
it is not the binding constraint. A hypothetical zero-step model (pure
noise passed straight to the VAE, not a real generation) would still
cost 27.57ms and consume 69% of the budget. **Optimizing step count
alone cannot make this fit 40ms; the fixed cost has to be the target,
not the loop.**

**The ceiling, reported because the prompt asked for it, secondary to
the point above:** no step count fits 40ms -- even 1 step (46.33ms)
misses by a wide-enough margin (6ms, several times the ±0.82ms noise
band) that this isn't a borderline case. 3 steps fits 100ms (84.42ms,
comfortably inside the ±2.65ms band); 4 narrowly misses (100.89ms,
outside 100ms by more than its own ±0.61ms noise band, so not noise).

**Against the pre-registered hypothesis (above): held, more strongly
than predicted.** VAE decode (~21.0ms) isn't just "comparable to" one
denoising step (~18-19ms) -- it's slightly *larger*. At N=1, VAE decode
alone is 45% of total frame time. The falsification condition (VAE
decode at 10-20% or less of one step, at any step count) is not close
to true anywhere in this table.

**Resolution-specific, stated plainly:** VAE decode cost scales with
output resolution (more pixels to decode); the 27.57ms / 68.9% figures
above are for 512x512 specifically, not a property of diffusion models
in general. A higher-resolution target would very likely make this
finding *more* true (VAE decode grows, conditioning stays roughly
fixed), not less -- but that's not measured here, and the numbers above
should not be read as resolution-independent.

## Phase 6: LPIPS ranked the more-degraded image as closer to baseline

The single most useful methodological result in this repo, not because
the metric is unusual (LPIPS is the standard choice -- the same family
DeepCache's and TeaCache's own papers use for exactly this comparison)
but because trusting it alone here would have produced the wrong
conclusion, silently.

`scripts/diffusion_quality_check.py` computed LPIPS distance between a
DeepCache-enabled frame and the same seed/steps generated with the
cache disabled, at two step counts (`results/h200/diffusion/
quality_steps4_lpips.json`, `quality_steps20_lpips.json`):

| Steps | LPIPS distance |
|---|---|
| 4 | 0.4801 |
| 20 | 0.6776 |

Read only as numbers, this says steps=4's cached output is *closer* to
its non-cached baseline than steps=20's is -- caching costs less
quality at the step count nearest the real budget than at the "quality"
step count. **That reading is backwards.** The images, pulled and
looked at before writing anything down (`results/h200/diffusion/
quality_steps{4,20}_{nocache,deepcache}.png`):

![steps=4, no cache](../results/h200/diffusion/quality_steps4_nocache.png)
![steps=4, DeepCache](../results/h200/diffusion/quality_steps4_deepcache.png)
![steps=20, no cache](../results/h200/diffusion/quality_steps20_nocache.png)
![steps=20, DeepCache](../results/h200/diffusion/quality_steps20_deepcache.png)

At steps=20, DeepCache changes the *composition* -- the non-cached
frame is a sharp, full-face portrait; the cached frame at the same
seed is a sharp close-up of an entirely different facial region. Both
images are individually coherent and detailed; they just aren't the
same picture. At steps=4, the non-cached frame is already low-detail
(4 steps from pure noise is close to the lower edge of what this
scheduler can render *at all*), and the cached frame at steps=4 has
lost nearly all remaining structure -- a soft, largely-featureless
blur. The steps=4 pair looks *more* alike only in the sense that two
blurry things resemble each other more than two sharp, different
things do.

**Mechanism:** LPIPS (like most learned perceptual metrics) scores
distance in a deep feature space tuned to detect texture- and
edge-level differences. Two low-detail images produce weak activations
in comparable regions of that feature space almost regardless of their
content, because there isn't much texture or edge structure in either
one to disagree about -- the metric reads "both are soft" as "these are
similar." Two sharp, structurally different images produce strong,
different activations, which the metric correctly reads as "these are
different" -- even though, for this project's purposes (does caching
preserve the intended output at a given step count), the steps=20 pair
is arguably the *less* concerning case: it's a wrong picture, not a
degraded one, at a step count nobody would actually ship at 25fps
anyway. The steps=4 pair is the one that matters, and LPIPS ranked it
as the *better*-preserved result.

**Stated plainly, because this is the point of writing the entry at
all: read as a standalone number, LPIPS would have supported exactly
the wrong conclusion here -- that DeepCache's quality cost is smaller
in the low-step regime the real-time budget actually forces, when the
opposite is true. The only reason this didn't ship as a finding is that
the images were pulled and looked at before the number was trusted.**
A metric was checked against ground truth (the actual pixels) rather
than reported on its own authority -- the same discipline this repo has
applied to every other number that turned out to need it (FP8's
sample text alongside its `finish_reason` counts; the actual decoded
tokens behind AWQ's perplexity delta).
**How to apply:** any future quality-cost claim in this repo that
relies on a learned perceptual metric gets a visual spot-check before
being trusted standalone, every time, not just when the number looks
surprising -- this is exactly the kind of failure that looks
unsurprising until someone looks.

## Phase 6: DeepCache, honestly scoped

`scripts/diffusion_sweep.sh`, `--cache-interval 5 --cache-branch 0`
(the DeepCache paper's commonly-cited SD1.5 default; not tuned per
step count -- see Limitations), 5 repeats per cell:

| Steps | total, no cache (ms) | total, DeepCache (ms) | speedup |
|---|---|---|---|
| 1 | 46.33 | 49.27 | **0.94x (slower)** |
| 2 | 64.88 | 49.41 | 1.31x |
| 3 | 84.42 | 50.71 | 1.66x |
| 4 | 100.89 | 52.04 | 1.94x |
| 5 | 132.42 | 54.57 | 2.43x |
| 8 | 173.63 | 79.83 | 2.18x |
| 12 | 241.76 | 105.27 | 2.30x |
| 20 | 390.99 | 138.75 | **2.82x** |

**Headline speedup (2.82x at steps=20) and the honest scope of it: at
the one step count (N=1) closest to the model actually meeting the
40ms budget, DeepCache is *slower*, not faster.** With `cache_interval
= 5` and only one step total, there is no later step to reuse a cached
computation from -- the caching machinery adds bookkeeping overhead
(a wrapped `unet.forward`, cache-hit checks on every block) with zero
opportunity to skip anything, a pure cost with no offsetting benefit.
Speedup only becomes clearly positive at N>=2 and only becomes large at
N>=5 -- past the point (N=3, see above) that fits even the loose 100ms
reference budget. **The standard acceleration technique for this class
of model provides its largest benefit in exactly the step-count regime
a real-time avatar's own budget rules out, and provides no benefit --
a regression -- in the regime the budget actually forces.**

**The speedup curve is not smooth, and that's a real, explicable
pattern, not noise:** 2.43x at N=5 vs. 2.18x at N=8 is a dip, not
monotonic growth, despite N=8 being deeper into "more steps means more
caching opportunity" territory. `cache_interval=5` means one step in
every five is "real" (full computation); how large a fraction of a
given N is real depends on N mod 5, not on N being larger. N=5: 1 real
step out of 5 (20%). N=8: 2 real steps out of 8 (steps 0 and 5 -- 25%).
A larger real-step fraction means less caching, means less speedup --
so N=8 having *more* real-step fraction than N=5 correctly predicts
N=8's *smaller* speedup, a sawtooth pattern following N mod
`cache_interval`, not a smooth function of N. Noise bands on every
cell above are small relative to this gap (largest total-time stdev in
either table is 3.27ms, at N=1 DeepCache, against gaps of tens of ms)
-- confirmed real, not confirmed by repeats alone, by having a
mechanism that predicts the direction of the dip in advance of reading
the numbers a second time.

**Quality cost:** see the LPIPS entry above -- severe at both step
counts checked (steps=4, closest to the real budget: near-total loss
of structure; steps=20, the "quality" regime: a different image, not a
degraded one). Not measured at every step count in the speedup table --
two points, chosen at the two ends of what's relevant here (the
budget-forced regime and the quality-focused regime), not a claim that
quality cost is characterized everywhere on this curve.
**Not tuned:** one fixed `(cache_interval, cache_branch)` setting was
used throughout, matching the project's own standing rule against
turning a knob and calling it optimization (the same reasoning that
kept this phase from being a step-count sweep in the first place -- see
the prompt this phase started from). A smaller interval might trade
back some of N=1-4's regression for a smaller quality cost at those
step counts; untested, a real limitation, not investigated further
here.
