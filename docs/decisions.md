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
