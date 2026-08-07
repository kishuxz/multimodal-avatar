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
