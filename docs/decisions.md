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
