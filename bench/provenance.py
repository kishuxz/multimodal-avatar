"""Provenance capture: every results/*.json embeds the output of capture().

Rule: no result is trustworthy without knowing exactly what produced it
(GPU, driver, CUDA, vLLM, model revision, repo state, CLI args). This is
the single place that knowledge lives so every script calls the same code.
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _git_sha():
    return _run(["git", "rev-parse", "HEAD"])


def _git_dirty():
    status = _run(["git", "status", "--porcelain"])
    return bool(status) if status is not None else None


def _gpu_info():
    csv = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    name, driver_version = None, None
    if csv:
        parts = [p.strip() for p in csv.split(",")]
        if len(parts) == 2:
            name, driver_version = parts

    # nvidia-smi's plain-text header reports the *max* CUDA version the
    # driver supports, not what actually ran the benchmark. The pinned
    # vLLM Docker image tag is the authoritative source for that (see
    # docs/decisions.md) -- this field is context, not ground truth.
    driver_max_cuda = None
    header = _run(["nvidia-smi"])
    if header:
        for line in header.splitlines():
            if "CUDA Version" in line:
                driver_max_cuda = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                break

    return {
        "name": name,
        "driver_version": driver_version,
        "driver_max_cuda_version": driver_max_cuda,
    }


def _vllm_version(vllm_server_url=None):
    """Prefer asking the running server (it's the code that actually served
    requests) over a local `import vllm`, which may not even be installed
    on the host driving the benchmark."""
    if vllm_server_url:
        try:
            req = urllib.request.Request(f"{vllm_server_url.rstrip('/')}/version")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.load(resp).get("version")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return None


def capture(
    model=None,
    model_revision=None,
    vllm_docker_image=None,
    vllm_server_url=None,
    extra=None,
):
    """Build the provenance block every results/*.json must include.

    Args:
        model: model name/path as passed to the server (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
        model_revision: HF revision/commit of that model.
        vllm_docker_image: pinned image tag the server was launched from.
        vllm_server_url: base URL of the running server, used to read its
            reported version instead of guessing from the local Python env.
        extra: dict of anything run-specific (arm name, load params, etc.)
            that doesn't belong in the fixed schema above.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_info(),
        "vllm_version": _vllm_version(vllm_server_url),
        "vllm_docker_image": vllm_docker_image,
        "model": model,
        "model_revision": model_revision,
        "python_version": sys.version.split()[0],
        "cli_args": sys.argv[1:],
        "extra": extra or {},
    }


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
