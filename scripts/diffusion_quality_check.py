"""
LPIPS perceptual distance between a DeepCache-enabled frame and the
same seed/steps/prompt generated without caching -- the quality cost of
the one optimization applied in Phase 6. Same metric family the
DeepCache/TeaCache papers themselves use for this comparison, not a
metric invented for this repo.

Non-cached output is treated as the reference, not "ground truth" --
there's no ground-truth image for a generative model at a given
seed/step count, only "what this same model/scheduler/seed produces
without the approximation being tested." LPIPS distance from that
reference is a real number regardless, and it's the number that answers
"does caching change the output," not a claim that the non-cached image
is uniquely correct.

Usage:
  python scripts/diffusion_quality_check.py \
      --reference results/h200/diffusion/quality_steps20_nocache.png \
      --candidate results/h200/diffusion/quality_steps20_deepcache.png \
      --out results/h200/diffusion/quality_steps20_lpips.json
"""
from __future__ import annotations

import argparse
import json
import os

import lpips
import torch
from PIL import Image


def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    import numpy as np
    arr = np.array(img).astype("float32") / 255.0  # HWC, [0, 1]
    arr = arr * 2 - 1  # LPIPS expects [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="non-cached image")
    p.add_argument("--candidate", required=True, help="deep-cache image, same seed/steps")
    p.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()

    loss_fn = lpips.LPIPS(net=cfg.net)
    ref = load_image_tensor(cfg.reference)
    cand = load_image_tensor(cfg.candidate)

    if ref.shape != cand.shape:
        raise RuntimeError(f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}")

    with torch.no_grad():
        distance = loss_fn(ref, cand).item()

    payload = {
        "reference": cfg.reference,
        "candidate": cfg.candidate,
        "net": cfg.net,
        "lpips_distance": distance,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
