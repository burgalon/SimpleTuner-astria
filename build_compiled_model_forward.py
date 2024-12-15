import torch
import os
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch.utils.benchmark as benchmark

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch.utils.benchmark as benchmark
from functools import partial
from pathlib import Path
from peft import LoraConfig


current_file = Path(__file__)
parent_directory = current_file.parent
absolute_parent_directory = parent_directory.resolve()

def prepare_latents(batch_size, height, width, num_channels_latents=1):
    vae_scale_factor = 16
    height = 2 * (int(height) // vae_scale_factor)
    width = 2 * (int(width) // vae_scale_factor)
    shape = (batch_size, num_channels_latents, height, width)
    pre_hidden_states = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    hidden_states = FluxPipeline._pack_latents(
        pre_hidden_states, batch_size, num_channels_latents, height, width
    )
    return hidden_states

def get_example_inputs(batch_size, height, width, num_channels_latents=1):
    hidden_states = prepare_latents(batch_size, height, width, num_channels_latents)
    num_img_sequences = hidden_states.shape[1]
    example_inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": torch.randn(batch_size, 512, 4096, dtype=torch.bfloat16, device="cuda"),
        "pooled_projections": torch.randn(batch_size, 768, dtype=torch.bfloat16, device="cuda"),
        "timestep": torch.tensor([1.0], device="cuda").expand(batch_size),
        "img_ids": torch.randn(num_img_sequences, 3, dtype=torch.bfloat16, device="cuda"),
        "txt_ids": torch.randn(512, 3, dtype=torch.bfloat16, device="cuda"),
        "guidance": torch.tensor([3.5],  device="cuda").expand(batch_size),
        "return_dict": False,
    }
    return example_inputs

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

def load_model():
    try:
        from flash_attn_interface import flash_attn_func
        import diffusers

        from helpers.models.flux.attention import (
            FluxAttnProcessor3_0,
            FluxSingleAttnProcessor3_0,
        )

        diffusers.models.attention_processor.FluxSingleAttnProcessor2_0 = (
            FluxSingleAttnProcessor3_0
        )
        diffusers.models.attention_processor.FluxAttnProcessor2_0 = (
            FluxAttnProcessor3_0
        )
        print("Using FlashAttention3_0 for H100 GPU (Single block)")
    except Exception as e:
        print(f"Can not use flash attn: {e}")
        print(
            "No flash_attn is available, using slower FlashAttention_2_0. Install flash_attn to make use of FA3 for Hopper or newer arch."
        )

    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    return model

def aot_compile(name, fn, absolute_parent_directory=absolute_parent_directory, **sample_kwargs):
    path = f"{absolute_parent_directory}/{name}.so"
    print(f"Storing compiled model to {path=}")
    options = {
        "aot_inductor.output_path": path,
        "max_autotune": True,
        "triton.cudagraphs": True,
    }

    torch._export.aot_compile(
        fn,
        (),
        sample_kwargs,
        options=options,
        disable_constraint_solver=True,
    )
    return path

def aot_load(path):
    return torch._export.aot_load(path, "cuda")

@torch.no_grad()
def f(model, **kwargs):
    return model(**kwargs)

model = load_model()
num_channels_latents = model.config.in_channels // 4 

# Validation/inference
inputs = get_example_inputs(
    batch_size=1,
    height=1024,
    width=1024,
    num_channels_latents=num_channels_latents,
)
# path = aot_compile(f"bs_1_{resolution}", partial(f, model=model), **inputs)

compiled_func_1 = aot_load('bs_1_1024.so')
print(f"{compiled_func_1(**inputs)[0].shape=}")

for _ in range(5):
    _ = compiled_func_1(**inputs)[0]

time = benchmark_fn(f, compiled_func_1, **inputs)
print('Batch size 1 time', time) # 0.421 seconds on an A100.

"""
transformer_lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "to_k",
        "to_q",
        "to_v",
        "add_k_proj",
        "add_q_proj",
        "add_v_proj",
        "to_out.0",
        "to_add_out",
    ],
    use_dora=False,
)
model.add_adapter(transformer_lora_config)

resolution = os.environ.get('RESOLUTION', '512')
resolution = int(resolution)

inputs = get_example_inputs(
    batch_size=4,
    height=resolution,
    width=resolution,
    num_channels_latents=num_channels_latents,
)
path = aot_compile(f"bs_4_{resolution}", partial(f, model=model), **inputs)

compiled_func_1 = aot_load(path)
print(f"{compiled_func_1(**inputs)[0].shape=}")

for _ in range(5):
    _ = compiled_func_1(**inputs)[0]

time = benchmark_fn(f, compiled_func_1, **inputs)
print('Batch size 4 time', time) # 0.421 seconds on an A100.
"""
