"""
One-time data-prep step for Phase 4 (perplexity): cut a fixed slice of the
wikitext-2-raw-v1 test set, tokenized with the model family's own tokenizer,
and commit both the token ids (what actually gets sent to the server) and
the decoded text (so a reviewer can read what's being evaluated without
running anything).

Not part of the pod pipeline -- runs once, locally, off-GPU, before any
vLLM server exists. Needs `pyarrow`, `transformers`, and `huggingface_hub`
locally; deliberately not added to requirements.txt, which pins what the
pod install needs, not this one-off tool (see docs/decisions.md).

Usage:
  python scripts/build_perplexity_slice.py \
      --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
      --n-tokens 8192 \
      --out-ids data/wikitext2_test_slice_token_ids.json \
      --out-text data/wikitext2_test_slice.txt
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
    p.add_argument("--n-tokens", type=int, default=8192)
    p.add_argument("--out-ids", default="data/wikitext2_test_slice_token_ids.json")
    p.add_argument("--out-text", default="data/wikitext2_test_slice.txt")
    return p.parse_args()


def main():
    cfg = parse_args()

    local_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    with open(local_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    table = pq.read_table(local_path)
    # Rows are already newline-terminated lines of the original wikitext
    # file (confirmed by inspection); plain concatenation reconstructs it,
    # no separator needed.
    full_text = "".join(table.column("text").to_pylist())

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    all_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(all_ids) < cfg.n_tokens:
        raise RuntimeError(
            f"wikitext-2 test set only tokenizes to {len(all_ids)} tokens "
            f"with {cfg.tokenizer}, less than requested --n-tokens {cfg.n_tokens}"
        )
    slice_ids = all_ids[: cfg.n_tokens]
    slice_text = tokenizer.decode(slice_ids)

    os.makedirs(os.path.dirname(cfg.out_ids) or ".", exist_ok=True)
    with open(cfg.out_ids, "w") as f:
        json.dump(
            {
                "source_repo": DATASET_REPO,
                "source_file": DATASET_FILE,
                "source_file_sha256": sha256,
                "tokenizer": cfg.tokenizer,
                "n_tokens": len(slice_ids),
                "token_ids": slice_ids,
            },
            f,
        )
    with open(cfg.out_text, "w") as f:
        f.write(slice_text)

    print(f"wrote {cfg.out_ids} ({len(slice_ids)} token ids)")
    print(f"wrote {cfg.out_text} ({len(slice_text)} chars, decoded from those ids)")
    print(f"source: {DATASET_REPO}/{DATASET_FILE}, sha256={sha256}")


if __name__ == "__main__":
    main()
