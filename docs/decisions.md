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

## Pod environment (Phase 1, first hardware step)

**Hardware:** RunPod, single H100 80GB HBM3. Driver `580.126.09`, host
CUDA 13.0. Container image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(ships torch 2.8.0+cu128 baseline). Network volume at `/workspace`.

**Chose:** `HF_HOME=/workspace/hf`, set in `~/.bashrc` so it survives
reconnects.
**Why:** container disk resets on pod stop; the network volume at
`/workspace` doesn't. Model weights are tens of GB -- re-downloading
them every time the pod restarts would be slow and wasteful, and
would silently change what's cached vs. what's cold on the next run.

**Found:** `pip install vllm` replaced the base image's torch entirely --
`torch 2.8.0+cu128` uninstalled, `torch 2.11.0+cu130` installed (pulled
in by `vllm==0.26.0`'s dependency resolution), along with
`torchvision 0.26.0` and `torchaudio 2.11.0`. `torch.version.cuda` moved
from `12.8` to `13.0` -- which happens to now match the host driver's
CUDA 13.0 more closely than the container's original build did.
**Why it matters:** every result file's provenance has to record the
*actual* installed torch/CUDA versions, not the container tag's implied
ones -- the tag name (`torch280-cu1281`) describes the base image, not
what's running once `pip install vllm` has had its way with it.
**Accelerator kernels:** no `flash-attn` package by that name;
`flashinfer-python 0.6.14` came along instead (vLLM's default attention
kernel backend on this version). No `xformers`.

Versions as installed:
- vLLM: `0.26.0`
- torch: `2.11.0+cu130`
- torch.version.cuda: `13.0`

**Chose:** connect over a direct TCP SSH endpoint (`root@<pod-ip>:<port>`,
normal exec-mode SSH) rather than RunPod's `ssh.runpod.io` proxy.
**Rejected:** the proxy as the primary connection.
**Why:** the proxy only supports interactive PTY sessions, not exec
mode -- every command has to go through a live shell via stdin, and the
PTY's line-editing occasionally mangles piped-in commands (observed:
a `kill -0 <pid>` liveness check silently corrupted, producing a false
"process finished" reading on a `pip install` that was still running).
Direct TCP gets a normal command channel: `ssh host 'cmd'` runs once and
returns, no output-parsing games. The proxy stays configured as a
fallback for when the pod's public IP isn't reachable.
**How to apply:** the direct endpoint's IP and port are assigned per pod
and change on every restart. Nothing in this repo hardcodes them --
scripts take the endpoint as a parameter or read it from the shell
environment at run time, never bake it into a committed file.

**Changed from the original plan:** serving via `pip install vllm` on
RunPod's own PyTorch base image, not the official vLLM OpenAI-compatible
Docker image the kickoff spec called for.
**Why:** not recorded yet -- this was a call made when setting up the
pod, and the reasoning behind it isn't captured here. Flagging rather
than inventing one.
