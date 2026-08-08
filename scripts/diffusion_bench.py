"""
Phase 6: per-frame stage timing for a diffusion model, as a proxy for a
real-time avatar's per-frame generation budget (40ms @ 25fps; 100ms as a
looser reference point). Not vLLM, not a server -- a plain Python process
running `diffusers` directly, timed stage by stage with
`torch.cuda.synchronize()` bracketing every stage. GPU work is
asynchronous; a CPU-side `time.perf_counter()` call with no synchronize
measures whatever happened to be queued, not what actually ran -- every
timed region here is synchronize -> perf_counter -> [stage] ->
synchronize -> perf_counter, so what's reported is real elapsed GPU time,
not queue-submission time. See docs/decisions.md for a worked example
(the pilot run) of why this matters: unsynchronized timing of the first
call to every stage reads 100-300ms of pure CUDA/cuDNN warmup as if it
were the stage's real cost.

Stages timed separately, every run:
  - conditioning : tokenize + CLIP text encode (cond + uncond)
  - step_i       : one denoising step (UNet forward, CFG combine,
                    scheduler.step), for each of --steps steps
  - vae_decode   : final latents -> pixel-space image

One full warmup generation is run and discarded before any timed run --
same discipline as scripts/calibrate.py / scripts/sweep.sh's --warmup,
applied here for the same reason (see docs/decisions.md).

Usage:
  python scripts/diffusion_bench.py --steps 4 --repeats 5 \
      --out results/h200/diffusion/steps4.json

  # DeepCache-enabled run, same step count, for the speedup/quality
  # comparison:
  python scripts/diffusion_bench.py --steps 20 --repeats 5 \
      --deep-cache --cache-interval 5 --cache-branch 0 \
      --out results/h200/diffusion/steps20_deepcache.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import provenance

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPT = "a portrait of a person speaking, photorealistic, studio lighting"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted"
GUIDANCE_SCALE = 7.5


def load_pipe(device):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_once(pipe, device, num_steps, height, width, seed, prompt, negative_prompt):
    """One full frame generation, every stage synchronized and timed
    separately. Returns (cond_time_s, [step_time_s, ...], vae_time_s, image)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        text_embeds = pipe.text_encoder(text_inputs.input_ids)[0]
        uncond_inputs = pipe.tokenizer(
            negative_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        uncond_embeds = pipe.text_encoder(uncond_inputs.input_ids)[0]
        embeds = torch.cat([uncond_embeds, text_embeds])
    torch.cuda.synchronize()
    t_cond = time.perf_counter() - t0

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator, device=device, dtype=torch.float16,
    )
    pipe.scheduler.set_timesteps(num_steps, device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    step_times = []
    with torch.no_grad():
        for t in pipe.scheduler.timesteps:
            torch.cuda.synchronize()
            ts0 = time.perf_counter()
            latent_input = torch.cat([latents] * 2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - ts0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        latents_scaled = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents_scaled).sample
    torch.cuda.synchronize()
    t_vae = time.perf_counter() - t0

    return t_cond, step_times, t_vae, image


def tensor_to_pil(image_tensor):
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    from PIL import Image
    import numpy as np
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--deep-cache", action="store_true")
    p.add_argument("--cache-interval", type=int, default=5)
    p.add_argument("--cache-branch", type=int, default=0)
    p.add_argument("--save-image", default=None, help="path to save the last repeat's decoded frame as PNG")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    device = "cuda"

    pipe = load_pipe(device)

    deep_cache_helper = None
    if cfg.deep_cache:
        from DeepCache import DeepCacheSDHelper
        deep_cache_helper = DeepCacheSDHelper(pipe=pipe)
        deep_cache_helper.set_params(cache_interval=cfg.cache_interval, cache_branch_id=cfg.cache_branch)
        deep_cache_helper.enable()

    # Warmup: one full generation, fully discarded. Not timed, not
    # recorded -- see module docstring / docs/decisions.md for why an
    # unwarmed first call isn't a real number.
    run_once(pipe, device, cfg.steps, cfg.height, cfg.width, seed=999999,
             prompt=cfg.prompt, negative_prompt=cfg.negative_prompt)

    per_repeat = []
    last_image = None
    for i in range(cfg.repeats):
        seed = cfg.seed_start + i
        t_cond, step_times, t_vae, image = run_once(
            pipe, device, cfg.steps, cfg.height, cfg.width, seed,
            cfg.prompt, cfg.negative_prompt)
        total = t_cond + sum(step_times) + t_vae
        last_image = image
        per_repeat.append({
            "seed": seed,
            "cond_s": t_cond,
            "step_s": step_times,
            "step_mean_s": statistics.fmean(step_times),
            "vae_s": t_vae,
            "total_s": total,
        })
        print(
            f"  repeat {i + 1}/{cfg.repeats} (seed={seed}): "
            f"cond={t_cond * 1000:.2f}ms step_mean={statistics.fmean(step_times) * 1000:.2f}ms "
            f"vae={t_vae * 1000:.2f}ms total={total * 1000:.2f}ms"
        )

    if cfg.save_image:
        os.makedirs(os.path.dirname(cfg.save_image) or ".", exist_ok=True)
        tensor_to_pil(last_image).save(cfg.save_image)
        print(f"  saved last frame -> {cfg.save_image}")

    conds = [r["cond_s"] for r in per_repeat]
    step_means = [r["step_mean_s"] for r in per_repeat]
    totals = [r["total_s"] for r in per_repeat]
    vaes = [r["vae_s"] for r in per_repeat]
    stats = {
        "cond_s": {"mean": statistics.fmean(conds), "stdev": statistics.stdev(conds) if len(conds) > 1 else None},
        "step_mean_s": {"mean": statistics.fmean(step_means), "stdev": statistics.stdev(step_means) if len(step_means) > 1 else None},
        "vae_s": {"mean": statistics.fmean(vaes), "stdev": statistics.stdev(vaes) if len(vaes) > 1 else None},
        "total_s": {"mean": statistics.fmean(totals), "stdev": statistics.stdev(totals) if len(totals) > 1 else None,
                     "min": min(totals), "max": max(totals)},
    }

    payload = {
        "provenance": provenance.capture(
            model=MODEL_ID,
            extra={
                "script": "diffusion_bench.py",
                "scheduler": "DPMSolverMultistepScheduler",
                "steps": cfg.steps,
                "height": cfg.height,
                "width": cfg.width,
                "guidance_scale": GUIDANCE_SCALE,
                "dtype": "float16",
                "deep_cache": cfg.deep_cache,
                "cache_interval": cfg.cache_interval if cfg.deep_cache else None,
                "cache_branch": cfg.cache_branch if cfg.deep_cache else None,
                "warmup_runs": 1,
            },
        ),
        "config": vars(cfg),
        "stats": stats,
        "per_repeat": per_repeat,
    }

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"steps": cfg.steps, "deep_cache": cfg.deep_cache, "stats": stats}, indent=2))
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
