"""
One-time data-prep step for Phase 4 (perplexity): cut several
non-overlapping, fixed-length slices of the wikitext-2-raw-v1 test set,
tokenized with the model family's own tokenizer -- the input to the
cross-slice noise-band measurement (scripts/perplexity_multislice.py).

A single slice, repeated, can only measure whether forced-decoding on one
fixed input is bit-reproducible (it is -- see docs/decisions.md). It
can't measure the uncertainty that actually applies to a perplexity
claim: how much the number moves across different text. Multiple
slices, each scored once, do that.

Not part of the pod pipeline -- runs once, locally, off-GPU, before any
vLLM server exists. Needs `pyarrow`, `transformers`, and `huggingface_hub`
locally; deliberately not added to requirements.txt, same reasoning as
scripts/build_perplexity_slice.py.

Usage:
  python scripts/build_perplexity_slices.py \
      --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
      --n-slices 8 --tokens-per-slice 8192 \
      --out-ids data/wikitext2_test_slices.json \
      --out-text data/wikitext2_test_slices.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DATASET_REPO = "Salesforce/wikitext"
DATASET_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-slices", type=int, default=8)
    p.add_argument("--tokens-per-slice", type=int, default=8192)
    p.add_argument("--out-ids", default="data/wikitext2_test_slices.json")
    p.add_argument("--out-text", default="data/wikitext2_test_slices.txt")
    return p.parse_args()


def main():
    cfg = parse_args()

    local_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    with open(local_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    table = pq.read_table(local_path)
    full_text = "".join(table.column("text").to_pylist())

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    all_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    n_needed = cfg.n_slices * cfg.tokens_per_slice
    if len(all_ids) < n_needed:
        raise RuntimeError(
            f"wikitext-2 test set only tokenizes to {len(all_ids)} tokens "
            f"with {cfg.tokenizer}, less than {cfg.n_slices} x "
            f"{cfg.tokens_per_slice} = {n_needed} needed for non-overlapping slices"
        )

    # Sequential, non-overlapping chunks -- slice i covers tokens
    # [i*L, (i+1)*L). Each slice is a genuinely different sample of text
    # (different articles, mostly -- wikitext-2's test set is many
    # concatenated articles), not a resample of the same content.
    slices = []
    for i in range(cfg.n_slices):
        start = i * cfg.tokens_per_slice
        end = start + cfg.tokens_per_slice
        ids = all_ids[start:end]
        slices.append({"index": i, "token_ids": ids, "text": tokenizer.decode(ids)})

    os.makedirs(os.path.dirname(cfg.out_ids) or ".", exist_ok=True)
    with open(cfg.out_ids, "w") as f:
        json.dump(
            {
                "source_repo": DATASET_REPO,
                "source_file": DATASET_FILE,
                "source_file_sha256": sha256,
                "tokenizer": cfg.tokenizer,
                "n_slices": cfg.n_slices,
                "tokens_per_slice": cfg.tokens_per_slice,
                "slices": [{"index": s["index"], "token_ids": s["token_ids"]} for s in slices],
            },
            f,
        )
    with open(cfg.out_text, "w") as f:
        for s in slices:
            f.write(f"=== slice {s['index']} ===\n{s['text']}\n\n")

    print(f"wrote {cfg.out_ids} ({cfg.n_slices} slices x {cfg.tokens_per_slice} tokens)")
    print(f"wrote {cfg.out_text} (decoded, for reading)")
    print(f"source: {DATASET_REPO}/{DATASET_FILE}, sha256={sha256}")


if __name__ == "__main__":
    main()
