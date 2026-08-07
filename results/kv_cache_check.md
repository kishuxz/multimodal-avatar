# KV cache arithmetic check (Phase 5, GPU-free)

## Per-token KV cache bytes

head_dim = hidden_size / num_attention_heads = 1536 / 12 = 128

bytes/token = num_hidden_layers x num_key_value_heads x head_dim x 2 (K and V) x dtype_bytes
            = 28 x 2 x 128 x 2 x 2
            = 28,672 bytes/token (28.0 KiB/token)

## Per-sequence, at max_position_embeddings (an upper bound, not a prediction)

bytes/sequence (full 32768-token context) = 28,672 x 32,768
                                         = 939,524,096 bytes
                                         = 0.8750 GiB = 0.9395 GB

**This is an upper bound on one fully-extended sequence, not a
steady-state usage prediction.** vLLM allocates KV cache in fixed-
size paged blocks and only reserves blocks for tokens a sequence
has actually generated, not its maximum possible length -- the
same distinction PagedAttention exists to make relative to naive
contiguous per-request allocation. The Phase 3 sweep's actual
requests (`max_tokens=80`, prompts drawn from
`harness.py`'s `USER_TURNS`) run far short of 32,768 tokens, so realized per-sequence usage in
that data is a small fraction of the figure above -- this number
answers "what's the ceiling," not "what did the sweep use."

## No committed value to check this against

Nothing in `results/*.json` records vLLM's own KV-cache-size
figure. `scripts/sweep.sh` only ever reads `/metrics` for
`vllm:kv_cache_usage_perc` (a percentage, to confirm a cold cache
before each run) -- never an absolute block or byte count -- and
the server's full startup log (where vLLM's own `GPU KV cache
size:` line lives) is written to the pod's local disk
(`/workspace/vllm_sweep_${label}.log`) and never copied into this
repo. So the arithmetic above has nothing to diff against right
now. See the GitHub issue opened alongside this script.

## Illustrative capacity (order of magnitude -- not a vLLM-equivalent number)

`--gpu-memory-utilization 0.9` on a single H100 80GB budgets 72.0 GiB total for weights + activations + KV cache. Subtracting a rough weight footprint per arm and dividing the remainder by the per-max-length-sequence figure above gives a ceiling on concurrent full-context sequences -- ignoring activation memory, CUDA graph capture, and framework overhead entirely, so treat this as "which order of magnitude," not a real capacity plan:

| Arm | Approx weight bytes | KV budget (GiB) | Ceiling: full-32k-context sequences |
|---|---|---|---|
| fp16 | 2.87 GiB | 69.1 | 79 |
| awq (int4) | 0.72 GiB | 71.3 | 81 |
| fp8 (weight-only) | 1.43 GiB | 70.6 | 80 |
