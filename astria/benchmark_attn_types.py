#!/usr/bin/env python3
"""
Benchmark FLUX + SageAttention vs FLUX + FlashAttention-3 (vLLM kernels)
- LoRA path: /data/models/1533312.safetensors
- Generates 4 images per variant, measures wall time
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from diffusers import DiffusionPipeline
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/data/models/1504944-flux1",
                   help="Local FLUX model path (same form you use in your pipeline)")
    p.add_argument("--lora_path", type=str, default="/data/models/1533312.safetensors")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--prompt", type=str, default="portrait photo, soft studio lighting, 50mm, high detail")
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--warmup", type=int, default=1, help="Warmup images per variant (not timed)")
    return p.parse_args()

# ---------- SageAttention wiring ----------
def apply_sage_attention(pipe):
    """
    Mirrors your setup_sage_attention: put Sage processors on flux blocks.
    """
    from sageattention import sageattn  # ensures the ext is actually present
    from helpers.models.flux.attention import FluxAttnProcessorSage

    attn_procs = {}
    # Same index ranges you used
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))

    for name, attn_processor in pipe.transformer.attn_processors.items():
        # name example: 'transformer_blocks.0.attn1.processor'
        layer_index = None
        m = None
        try:
            import re
            m = re.search(r"\.(\d+)\.", name)
        except Exception:
            pass
        if m:
            layer_index = int(m.group(1))

        if name.startswith("transformer_blocks") and (layer_index in double_blocks_idx):
            attn_procs[name] = FluxAttnProcessorSage()
        elif name.startswith("single_transformer_blocks") and (layer_index in single_blocks_idx):
            attn_procs[name] = FluxAttnProcessorSage()
        else:
            attn_procs[name] = attn_processor

    pipe.transformer.set_attn_processor(attn_procs)

# ---------- FA3 wiring (vLLM kernels) ----------
def build_fa3_processor():
    try:
        from kernels import get_kernel
        _k = get_kernel("kernels-community/vllm-flash-attn3")
        _flash_attn_func = _k.flash_attn_func
        _kernels_err = None
    except Exception as e:
        _flash_attn_func, _kernels_err = None, e

    def _ensure_fa3_available():
        if _flash_attn_func is None:
            raise ImportError(
                "FlashAttention-3 via HF `kernels` is required but unavailable.\n"
                f"get_kernel('kernels-community/vllm-flash-attn3') failed with:\n{_kernels_err}"
            )

    @torch.library.custom_op("flash::flash_attn_func", mutates_args=())
    def flash_attn_func(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        out, _lse = _flash_attn_func(q, k, v, causal=causal)
        return out

    @flash_attn_func.register_fake
    def _(q: Tensor, k: Tensor, v: Tensor, causal: bool = False, **kwargs) -> Tensor:
        return torch.empty_like(q).contiguous()

    class FluxAttnProcessorFA3:
        _attention_backend = "fa3"

        def __init__(self):
            _ensure_fa3_available()
            build_fa3_processor._op_registered = True

        @torch.no_grad()
        def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: torch.FloatTensor = None,
            image_rotary_emb=None,
            **kwargs,
        ):
            if attention_mask is not None:
                raise NotImplementedError("FA3 path does not support arbitrary attention_mask.")

            B, S_img, D = hidden_states.shape
            H = getattr(attn, "heads", None) or getattr(attn, "num_heads", None)
            dh = D // H

            # (B, S, D) -> (B, H, S, Dh)
            def pack_hs(t):
                return t.view(B, -1, H, dh).transpose(1, 2).contiguous()

            # (B, H, S, Dh) -> (B, S, H, Dh) for FA3
            def to_fa3(t):
                return t.transpose(1, 2).contiguous()

            def maybe(t, fn):
                return fn(t) if fn is not None else t

            # ---------- SINGLE BLOCK ----------
            if encoder_hidden_states is None:
                q_hs = pack_hs(attn.to_q(hidden_states))
                k_hs = pack_hs(attn.to_k(hidden_states))
                v_hs = pack_hs(attn.to_v(hidden_states))

                q_hs = maybe(q_hs, getattr(attn, "norm_q", None))
                k_hs = maybe(k_hs, getattr(attn, "norm_k", None))

                if image_rotary_emb is not None:
                    from diffusers.models.embeddings import apply_rotary_emb
                    q_hs = apply_rotary_emb(q_hs, image_rotary_emb)
                    k_hs = apply_rotary_emb(k_hs, image_rotary_emb)

                out = flash_attn_func(to_fa3(q_hs), to_fa3(k_hs), to_fa3(v_hs), causal=False)  # (B,S,H,Dh)
                out = out.flatten(2, 3).to(hidden_states.dtype)  # (B,S,D)
                return out

            # ---------- DOUBLE BLOCK ----------
            # image projections (B,H,S,Dh)
            q_img_hs = pack_hs(attn.to_q(hidden_states))
            k_img_hs = pack_hs(attn.to_k(hidden_states))
            v_img_hs = pack_hs(attn.to_v(hidden_states))
            q_img_hs = maybe(q_img_hs, getattr(attn, "norm_q", None))
            k_img_hs = maybe(k_img_hs, getattr(attn, "norm_k", None))

            # context projections (added) (B,H,S_ctx,Dh)
            if hasattr(attn, "to_added_qkv"):
                enc_qkv = attn.to_added_qkv(encoder_hidden_states)
                split = enc_qkv.shape[-1] // 3
                enc_q_lin, enc_k_lin, enc_v_lin = torch.split(enc_qkv, split, dim=-1)
            else:
                enc_q_lin = attn.add_q_proj(encoder_hidden_states)
                enc_k_lin = attn.add_k_proj(encoder_hidden_states)
                enc_v_lin = attn.add_v_proj(encoder_hidden_states)

            q_ctx_hs = pack_hs(enc_q_lin)
            k_ctx_hs = pack_hs(enc_k_lin)
            v_ctx_hs = pack_hs(enc_v_lin)
            q_ctx_hs = maybe(q_ctx_hs, getattr(attn, "norm_added_q", None))
            k_ctx_hs = maybe(k_ctx_hs, getattr(attn, "norm_added_k", None))

            S_ctx = q_ctx_hs.shape[2]

            # concat along sequence in (B,H,S,Dh)
            q_cat_hs = torch.cat([q_ctx_hs, q_img_hs], dim=2)
            k_cat_hs = torch.cat([k_ctx_hs, k_img_hs], dim=2)
            v_cat_hs = torch.cat([v_ctx_hs, v_img_hs], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                q_cat_hs = apply_rotary_emb(q_cat_hs, image_rotary_emb)
                k_cat_hs = apply_rotary_emb(k_cat_hs, image_rotary_emb)

            out_cat = flash_attn_func(to_fa3(q_cat_hs), to_fa3(k_cat_hs), to_fa3(v_cat_hs), causal=False)  # (B,S_tot,H,Dh)
            out_ctx = out_cat[:, :S_ctx, :, :].flatten(2, 3).to(hidden_states.dtype)  # (B,S_ctx,D)
            out_img = out_cat[:, S_ctx:, :, :].flatten(2, 3).to(hidden_states.dtype)  # (B,S_img,D)

            # project like Sage
            if hasattr(attn, "to_out"):
                out_img = attn.to_out[0](out_img)
                if len(attn.to_out) > 1:
                    out_img = attn.to_out[1](out_img)
            if hasattr(attn, "to_add_out"):
                out_ctx = attn.to_add_out(out_ctx)

            return out_img, out_ctx

    return FluxAttnProcessorFA3


def apply_fa3_attention(pipe):
    FluxAttnProcessorFA3 = build_fa3_processor()
    attn_procs = {name: FluxAttnProcessorFA3() for name in pipe.transformer.attn_processors.keys()}
    pipe.transformer.set_attn_processor(attn_procs)

# ---------- Helpers ----------
def load_flux(model_path: str, dtype: str):
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to("cuda")
    return pipe

def load_lora(pipe, lora_path: str, scale: float, adapter_name: str):
    print(f"-> Loading LoRA: {lora_path} (adapter_name={adapter_name}, scale={scale})")
    pipe.load_lora_weights(lora_path, adapter_name=adapter_name, low_cpu_mem_usage=False)
    pipe.set_adapters([adapter_name], adapter_weights=[scale])

@torch.inference_mode()
def generate_n(pipe, prompt, negative_prompt, W, H, steps, n, seed, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    if hasattr(pipe.transformer, "set_number_of_steps"):
        pipe.transformer.set_number_of_steps(steps)

    # Pre-encode prompts (matches your pipeline style)
    enc = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,  # FLUX dual encoders
        max_sequence_length=512,
        device="cuda",
    )
    prompt_embeds, pooled_prompt_embeds, _ = enc

    kwargs = dict(
        height=H,
        width=W,
        num_inference_steps=steps,
        guidance_scale=3.5,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )

    for i in range(n):
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        if hasattr(pipe.transformer, "clear_cache"):
            pipe.transformer.clear_cache()
        img = pipe(generator=g, **kwargs).images[0]
        images.append(img)
        img.save(outdir / f"{prefix}_{i:02d}.png")
    return images

def timed_run(pipe, *gen_args, **gen_kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    imgs = generate_n(pipe, *gen_args, **gen_kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return imgs, t1 - t0

# ---------- Main ----------
def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    outdir = Path(args.outdir)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU_MEMORY_GB={total_mem_gb:.1f}")

    # ========= Variant 1: SageAttention =========
    print("\n=== FLUX + SageAttention ===")
    pipe = load_flux(args.model_path, args.dtype)
    apply_sage_attention(pipe)
    load_lora(pipe, args.lora_path, args.lora_scale, adapter_name="bench_lora")

    # Warmup (not timed)
    if args.warmup > 0:
        _ = generate_n(pipe, args.prompt, args.negative_prompt, args.width, args.height,
                       args.steps, args.warmup, args.seed, outdir, prefix="sage_warmup")

    # Timed run
    _, sage_secs = timed_run(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.width,
        args.height,
        args.steps,
        args.num_images,
        args.seed,
        outdir,
        prefix="sage",
    )
    print(f"SageAttention: {args.num_images} images in {sage_secs:.2f}s "
          f"({sage_secs/args.num_images:.2f}s/img)")

    # Cleanup between variants
    del pipe
    torch.cuda.empty_cache()

    # ========= Variant 2: FlashAttention-3 via kernels =========
    print("\n=== FLUX + FlashAttention-3 (vLLM kernels) ===")
    pipe = load_flux(args.model_path, args.dtype)
    apply_fa3_attention(pipe)
    load_lora(pipe, args.lora_path, args.lora_scale, adapter_name="bench_lora")

    # Warmup (not timed)
    if args.warmup > 0:
        _ = generate_n(pipe, args.prompt, args.negative_prompt, args.width, args.height,
                       args.steps, args.warmup, args.seed, outdir, prefix="fa3_warmup")

    # Timed run
    _, fa3_secs = timed_run(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.width,
        args.height,
        args.steps,
        args.num_images,
        args.seed,
        outdir,
        prefix="fa3",
    )
    print(f"FlashAttn-3: {args.num_images} images in {fa3_secs:.2f}s "
          f"({fa3_secs/args.num_images:.2f}s/img)")

    print("\n=== Summary ===")
    print(f"SageAttention total: {sage_secs:.2f}s\tper image: {sage_secs/args.num_images:.2f}s")
    print(f"FlashAttn-3 total:  {fa3_secs:.2f}s\tper image: {fa3_secs/args.num_images:.2f}s")
    if fa3_secs < sage_secs:
        print(f"Speedup (FA3 vs Sage): {sage_secs/fa3_secs:.2f}×")
    else:
        print(f"Speedup (Sage vs FA3): {fa3_secs/sage_secs:.2f}×")

if __name__ == "__main__":
    main()
