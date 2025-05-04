import os
import re
import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

import sys

# one level app + /astria
sys.path.append("/root/SimpleTuner-astria")

try:
    from sageattention import sageattn
    import diffusers

    from helpers.models.flux.attention import (
        FluxAttnProcessorSage,
        FluxSingleAttnProcessorSage,
        FusedFluxAttnProcessorSage,
    )

    # # override the 2.0 attention processor classes
    # diffusers.models.attention_processor.FluxSingleAttnProcessor2_0 = FluxSingleAttnProcessorSage
    # diffusers.models.attention_processor.FluxAttnProcessor2_0       = FluxAttnProcessorSage
    # diffusers.models.attention_processor.FusedFluxAttnProcessor2_0  = FusedFluxAttnProcessorSage

    print("✅ SageAttention processors installed")
except Exception as e:
    print(f"⚠️  Could not install SageAttention: {e}")
    raise e

MODEL_ID = "black-forest-labs/FLUX.1-dev"
OUTPUT_ROOT = "/data/models"
PROMPT = "a serene forest at sunrise, ultra realistic"

def setup_sage_attention(pipe):
    attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))

    for name, attn_processor in pipe.transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            attn_procs[name] = FluxAttnProcessorSage()
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            attn_procs[name] = FluxAttnProcessorSage()
        else:
            attn_procs[name] = attn_processor

    pipe.transformer.set_attn_processor(attn_procs)

def setup_sage_attention_fused(pipe):
    attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))

    for name, attn_processor in pipe.transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            attn_procs[name] = FusedFluxAttnProcessorSage()
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            attn_procs[name] = FusedFluxAttnProcessorSage()
        else:
            attn_procs[name] = attn_processor

    pipe.transformer.set_attn_processor(attn_procs)

def load_pipeline(use_sage: bool, use_sage_fused: bool = False):
    print(f"\nLoading pipeline (use_sage={use_sage})…")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to("cuda")
    if use_sage and not use_sage_fused:
        setup_sage_attention(pipe)
    if use_sage and use_sage_fused:
        setup_sage_attention_fused(pipe)
    # make sure all GPU kernels are ready
    torch.cuda.synchronize()
    return pipe

def benchmark(pipe, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # single warm-up (to load kernels, caches, etc)
    _ = pipe(PROMPT, num_inference_steps=28, guidance_scale=7.5)
    torch.cuda.synchronize()

    # measured run
    start = time.time()
    generator = torch.Generator().manual_seed(1234)
    outputs = pipe(PROMPT, num_inference_steps=28, guidance_scale=7.5, generator=generator)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # save results
    for idx, img in enumerate(outputs.images):
        img.save(os.path.join(out_dir, f"out_{idx}.png"))

    print(f"→ Inference took {elapsed:.2f}s, saved to {out_dir}/")
    return elapsed

if __name__ == "__main__":
    results = {}

    # 1) Without SageAttention
    pipe = load_pipeline(use_sage=False)
    results["no_sage"] = benchmark(pipe, os.path.join(OUTPUT_ROOT, "benchmark_without_sage"))

    # 2) With SageAttention
    pipe = load_pipeline(use_sage=True)
    results["with_sage"] = benchmark(pipe, os.path.join(OUTPUT_ROOT, "benchmark_with_sage"))

    # 3) Fused
    # pipe = load_pipeline(use_sage=True, use_sage_fused=True)
    # results["with_sage_fused"] = benchmark(pipe, os.path.join(OUTPUT_ROOT, "benchmark_with_sage_fused"))

    # 4) Summary
    print("\n=== Summary ===")
    print(f"No SageAttention : {results['no_sage']:.2f}s")
    print(f"With SageAttention: {results['with_sage']:.2f}s")
    # print(f"With SageAttention Fused: {results['with_sage_fused']:.2f}s")
    speedup = results["no_sage"] / results["with_sage"] if results["with_sage"] > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}×")
    # speedup = results["no_sage"] / results["with_sage_fused"] if results["with_sage_fused"] > 0 else float("inf")
    # print(f"Speedup (fused): {speedup:.2f}×")
