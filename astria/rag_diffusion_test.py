import torch
import argparse
import json
from diffusers import FluxPipeline
from ragdiffusion import (
    RAG_FluxPipeline,
    RAG_FluxTransformer2DModel,
    openai_gpt4o_get_regions,
)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

pipe = RAG_FluxPipeline(
    scheduler=pipe.scheduler,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer_2=pipe.tokenizer_2,
    vae=pipe.vae,
    transformer=pipe.transformer,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help="Image prompt")
    parser.add_argument('--hb_replace', type=int, default=2, help="HB replace")
    parser.add_argument('--sr_delta', type=float, default=1.0, help="SR delta")
    return parser.parse_args()

args = parse_arguments()

regions = openai_gpt4o_get_regions(args.prompt)

prompt = args.prompt
HB_replace = args.hb_replace
HB_prompt_list =  regions["HB_prompt_list"]
HB_m_offset_list = regions["HB_m_offset_list"]
HB_n_offset_list = regions["HB_n_offset_list"]
HB_m_scale_list = regions["HB_m_scale_list"]
HB_n_scale_list = regions["HB_n_scale_list"]
SR_delta = args.sr_delta
SR_hw_split_ratio = regions["SR_hw_split_ratio"]
SR_prompt = regions["SR_prompt"]

height = 1024
width = 1024
seed = 12345


image = pipe(
    SR_delta=SR_delta,
    SR_hw_split_ratio=SR_hw_split_ratio,
    SR_prompt=SR_prompt,
    HB_prompt_list=HB_prompt_list,
    HB_m_offset_list=HB_m_offset_list,
    HB_n_offset_list=HB_n_offset_list,
    HB_m_scale_list=HB_m_scale_list,
    HB_n_scale_list=HB_n_scale_list,
    HB_replace=HB_replace,
    seed=seed,
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=20,
    guidance_scale=3.5,
).images[0]

filename = "RAG.png"
image.save(filename)
print(f"Image saved as {filename}")