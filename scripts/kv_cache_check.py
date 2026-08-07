"""
Expected KV cache memory per token/sequence, computed from Qwen2.5-1.5B-
Instruct's own published config -- no server, no GPU, no live vLLM
process involved.

Model constants below come from Qwen/Qwen2.5-1.5B-Instruct's config.json
(hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
max_position_embeddings, torch_dtype) -- the same 28-layer/2-KV-head/GQA
numbers docs/decisions.md already states from the same source. head_dim
isn't in config.json directly; it's hidden_size / num_attention_heads,
computed below rather than hardcoded a second time.

This is an EXPECTED figure, not a verified one: nothing in results/*.json
or anywhere else in this repo records what vLLM itself allocated for the
KV cache on the pod that produced the Phase 3 sweep. sweep.sh tees the
server's stdout (which carries vLLM's own "GPU KV cache size" log line)
to a per-run file on the pod's local disk, and that file is neither
copied into results/ nor committed -- see the GitHub issue opened
alongside this script. So there is currently nothing in this repo to
diff the number below against. That gap is the finding this script
surfaces, not a bug in the arithmetic.

Usage:
  python scripts/kv_cache_check.py
  (pure computation, writes results/kv_cache_check.md)
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Qwen/Qwen2.5-1.5B-Instruct config.json.
NUM_HIDDEN_LAYERS = 28
HIDDEN_SIZE = 1536
NUM_ATTENTION_HEADS = 12
NUM_KEY_VALUE_HEADS = 2  # GQA: 12 query heads share 2 KV heads, 6:1
MAX_POSITION_EMBEDDINGS = 32768  # vLLM's max_model_len default absent
                                  # --max-model-len -- sweep.sh never
                                  # passes that flag (scripts/sweep.sh)
KV_CACHE_DTYPE_BYTES = 2  # bf16 -- torch_dtype in config.json, and vLLM's
                            # KV cache dtype defaults to the model dtype
                            # absent --kv-cache-dtype, which sweep.sh also
                            # never passes

GIB = 1024 ** 3
GB = 1_000_000_000


def head_dim():
    assert HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0
    return HIDDEN_SIZE // NUM_ATTENTION_HEADS


def bytes_per_token():
    # layers x KV heads x head_dim x 2 (K and V) x dtype bytes
    return (NUM_HIDDEN_LAYERS * NUM_KEY_VALUE_HEADS * head_dim()
            * 2 * KV_CACHE_DTYPE_BYTES)


def bytes_per_full_context_sequence():
    return bytes_per_token() * MAX_POSITION_EMBEDDINGS


# Rough per-arm weight footprint, for the illustrative capacity estimate
# only -- param count x bytes/param, no accounting for embeddings being
# untied, activation memory, CUDA graphs, or framework overhead. This is
# NOT the number vLLM's own allocator would report; it exists to turn
# "0.94 GB per max-length sequence" into an order-of-magnitude "how many
# of those fit," not to replace the real log line the GitHub issue asks
# sweep.sh to start capturing.
QWEN25_1_5B_PARAMS = 1_540_000_000
ARM_WEIGHT_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "awq (int4)": 0.5,
    "fp8 (weight-only)": 1.0,
}
GPU_MEMORY_UTIL = 0.9
H100_80GB_BYTES = 80 * GIB


def derivation_text():
    hd = head_dim()
    bpt = bytes_per_token()
    bpseq = bytes_per_full_context_sequence()
    lines = [
        "## Per-token KV cache bytes",
        "",
        f"head_dim = hidden_size / num_attention_heads = "
        f"{HIDDEN_SIZE} / {NUM_ATTENTION_HEADS} = {hd}",
        "",
        f"bytes/token = num_hidden_layers x num_key_value_heads x head_dim "
        f"x 2 (K and V) x dtype_bytes",
        f"            = {NUM_HIDDEN_LAYERS} x {NUM_KEY_VALUE_HEADS} x {hd} x 2 x {KV_CACHE_DTYPE_BYTES}",
        f"            = {bpt:,} bytes/token ({bpt / 1024:.1f} KiB/token)",
        "",
        "## Per-sequence, at max_position_embeddings (an upper bound, not a prediction)",
        "",
        f"bytes/sequence (full {MAX_POSITION_EMBEDDINGS}-token context) = "
        f"{bpt:,} x {MAX_POSITION_EMBEDDINGS:,}",
        f"                                         = {bpseq:,} bytes",
        f"                                         = {bpseq / GIB:.4f} GiB "
        f"= {bpseq / GB:.4f} GB",
        "",
        "**This is an upper bound on one fully-extended sequence, not a",
        "steady-state usage prediction.** vLLM allocates KV cache in fixed-",
        "size paged blocks and only reserves blocks for tokens a sequence",
        "has actually generated, not its maximum possible length -- the",
        "same distinction PagedAttention exists to make relative to naive",
        "contiguous per-request allocation. The Phase 3 sweep's actual",
        "requests (`max_tokens=80`, prompts drawn from",
        "`harness.py`'s `USER_TURNS`) run far short of "
        f"{MAX_POSITION_EMBEDDINGS:,} tokens, so realized per-sequence usage in",
        "that data is a small fraction of the figure above -- this number",
        "answers \"what's the ceiling,\" not \"what did the sweep use.\"",
        "",
        "## No committed value to check this against",
        "",
        "Nothing in `results/*.json` records vLLM's own KV-cache-size",
        "figure. `scripts/sweep.sh` only ever reads `/metrics` for",
        "`vllm:kv_cache_usage_perc` (a percentage, to confirm a cold cache",
        "before each run) -- never an absolute block or byte count -- and",
        "the server's full startup log (where vLLM's own `GPU KV cache",
        "size:` line lives) is written to the pod's local disk",
        "(`/workspace/vllm_sweep_${label}.log`) and never copied into this",
        "repo. So the arithmetic above has nothing to diff against right",
        "now. See the GitHub issue opened alongside this script.",
    ]
    return "\n".join(lines)


def illustrative_capacity_text():
    budget = GPU_MEMORY_UTIL * H100_80GB_BYTES
    bpseq = bytes_per_full_context_sequence()
    lines = [
        "## Illustrative capacity (order of magnitude -- not a vLLM-equivalent number)",
        "",
        f"`--gpu-memory-utilization {GPU_MEMORY_UTIL}` on a single H100 80GB "
        f"budgets {budget / GIB:.1f} GiB total for weights + activations + "
        "KV cache. Subtracting a rough weight footprint per arm and dividing "
        "the remainder by the per-max-length-sequence figure above gives a "
        "ceiling on concurrent full-context sequences -- ignoring activation "
        "memory, CUDA graph capture, and framework overhead entirely, so "
        "treat this as \"which order of magnitude,\" not a real capacity plan:",
        "",
        "| Arm | Approx weight bytes | KV budget (GiB) | Ceiling: full-32k-context sequences |",
        "|---|---|---|---|",
    ]
    for arm, bytes_per_param in ARM_WEIGHT_BYTES_PER_PARAM.items():
        weight_bytes = QWEN25_1_5B_PARAMS * bytes_per_param
        kv_budget = budget - weight_bytes
        ceiling = int(kv_budget // bpseq)
        lines.append(
            f"| {arm} | {weight_bytes / GIB:.2f} GiB | {kv_budget / GIB:.1f} | {ceiling:,} |"
        )
    return "\n".join(lines)


def main():
    text = derivation_text() + "\n\n" + illustrative_capacity_text() + "\n"
    print(text)

    out_path = os.path.join(REPO_ROOT, "results", "kv_cache_check.md")
    with open(out_path, "w") as f:
        f.write("# KV cache arithmetic check (Phase 5, GPU-free)\n\n" + text)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
