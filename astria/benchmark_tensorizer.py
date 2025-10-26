#!/usr/bin/env python3
# bench_flux_load.py
#
# Usage:
#   python bench_flux_load.py /path/to/model --runs 10
#   # e.g. python bench_flux_load.py /data/models/1504944-flux1 --vae-fp32
#
# Notes:
#   - Expects your `tensorize.py` to be importable (same dir or on PYTHONPATH).
#   - First builds the tensorizer bundle (warm-up, not timed), then times 10 loads
#     using the bundle vs standard Diffusers.from_pretrained.
#   - Interleaves runs (A,B,A,B,…) to reduce OS cache bias.

import argparse
import gc
import os
import random
import statistics
import time
from typing import Callable, List, Tuple

import torch
from diffusers import DiffusionPipeline

# import your drop-in
from tensorize import (
    load_or_tensorize_bundle,
    load_bundle,
    bundle_exists,
    tensorize_pipeline,
)

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def _teardown(obj):
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    _sync()

def _time_once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    obj = fn()
    _sync()  # make sure CUDA ops complete before stopping timer
    dt = time.perf_counter() - t0
    _teardown(obj)
    return dt

def _fmt_stats(samples: List[float]) -> str:
    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return f"{mean:.2f}s ± {stdev:.2f}s (min {min(samples):.2f}s, max {max(samples):.2f}s)"

def benchmark(model_dir: str, runs: int, dtype: str, vae_fp32: bool, local_only: bool) -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[dtype]

    # --- Prepare tensorizer bundle (warm-up, not timed) ---
    # (If bundle not present, build once; then do a single “ensure it loads” call.)
    # if not bundle_exists(model_dir):
    start = time.time()
    print("⏳ Building tensorizer bundle (one-time)...")
    tensorize_pipeline(model_dir, force=False, torch_dtype=torch_dtype)
    print(f"tensorizer bundle serialized (one-time) in {start - time.time()} seconds...")

    # warm-up load to make sure everything is valid (not timed)
    _teardown(
        load_or_tensorize_bundle(
            model_dir,
            device=device,
            default_dtype=torch_dtype,
            dtype_overrides={"vae": torch.float32} if vae_fp32 else None,
        )
    )

    # --- Define loaders for the timed runs ---
    def load_hf():
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            local_files_only=local_only,
            torch_dtype=torch_dtype,
        ).to(device)
        return pipe

    def load_tensorizer():
        # Use fast-path loader against the already-built bundle
        pipe = load_bundle(
            model_dir,
            device=device,
            dtype_overrides={"vae": torch.float32} if vae_fp32 else None,
        )
        return pipe

    # --- Interleave runs to reduce cache bias ---
    order: List[Tuple[str, Callable[[], object]]] = [("hf", load_hf), ("tz", load_tensorizer)] * runs
    # Trim to exactly 2*runs and optionally shuffle within pairs to avoid fixed position effects
    order = order[: 2 * runs]

    hf_times: List[float] = []
    tz_times: List[float] = []

    print(f"🧪 Benchmarking {runs} runs each (interleaved). Device={device.type}, dtype={dtype}"
          f"{', VAE=fp32' if vae_fp32 else ''}")
    for i, (tag, loader) in enumerate(order, 1):
        dt = _time_once(loader)
        (hf_times if tag == "hf" else tz_times).append(dt)
        label = "HF  " if tag == "hf" else "TZ  "
        print(f"{i:2d}. {label} {dt:.2f}s")

    # --- Stats ---
    print("\n=== Results ===")
    print(f"HF (Diffusers.from_pretrained): { _fmt_stats(hf_times) }")
    print(f"TZ (Tensorizer bundle load)   : { _fmt_stats(tz_times) }")

    mean_hf = statistics.mean(hf_times)
    mean_tz = statistics.mean(tz_times)
    speedup = mean_hf / mean_tz if mean_tz > 0 else float("inf")
    print(f"\nSpeedup (HF / TZ): {speedup:.2f}×")

def main():
    p = argparse.ArgumentParser(description="Benchmark Diffusers vs Tensorizer load times")
    p.add_argument("model_dir", type=str, help="Local model directory (HF snapshot) or HF ID")
    p.add_argument("--runs", type=int, default=10, help="Runs per method (default: 10)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Torch dtype for loading")
    p.add_argument("--vae-fp32", action="store_true", help="Force VAE to load in fp32 (common for better image quality)")
    p.add_argument("--no-local", action="store_true", help="Allow network fetches (default is offline/local only)")
    args = p.parse_args()

    benchmark(
        model_dir=args.model_dir,
        runs=args.runs,
        dtype=args.dtype,
        vae_fp32=args.vae_fp32,
        local_only=not args.no_local,
    )

if __name__ == "__main__":
    main()
