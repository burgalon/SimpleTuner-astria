import gc
import os
import random

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers import FluxPipeline
from diffusers.utils import (
    make_image_grid,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    from pipeline_multi import FluxPipelineMultiPersonWithPulID
    from pulid_ext import PuLID

    base_model = "black-forest-labs/FLUX.1-dev"
    location = "/home/user/storage/hf_cache"

    pipe = FluxPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        cache_dir=location,
    )

    pulid_model = PuLID(local_dir='/home/user/storage/astria/data/cache/pulid')

    pipe = FluxPipelineMultiPersonWithPulID(
        scheduler=pipe.scheduler,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer_2=pipe.tokenizer_2,
        vae=pipe.vae,
        transformer=pipe.transformer,
        pulid=pulid_model,
    ).to('cuda')

    prompts = [
        (
            "A cheerful photograph captures the man and woman smiling warmly in a lush tropical oasis, surrounded by vibrant greenery and colorful flowers. The man, with his clean-shaven head, symmetrical features, and relaxed demeanor, stands beside the woman, whose dark hair is neatly pulled back, highlighting her almond-shaped eyes and soft, natural smile. Their genuine expressions exude a sense of harmony and connection, perfectly complementing the tranquil beauty of the tropical backdrop.",
            (1024, 1024),
        ),
    ]

    pulid_scales_to_test = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    def add_text_to_image(img, text):
        draw = ImageDraw.Draw(img)
        # Load a font (TrueType or OpenType) - adjust path to your font file
        font = ImageFont.load_default(size=48)
        # Position for the text - adjust as needed
        text_position = (10, 10)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
        return img

    INFERENCE_PATH = 'test/test_inference'
    os.makedirs(INFERENCE_PATH, exist_ok=True)

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using CUDA
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(12345)

    IMAGE_FOLDER_1 = "../astria_tests/images/1815518"
    IMAGE_FOLDER_2 = "../astria_tests/images/1858416"

    def load_images(image_folder: str) -> list[Image.Image]:
        imgs = []
        for img in os.listdir(image_folder):
            imgs.append(Image.open(f'{image_folder}/{img}'))
        return imgs

    with torch.no_grad(), torch.inference_mode():
        subject_1 = load_images(IMAGE_FOLDER_1)
        subject_2 = load_images(IMAGE_FOLDER_2)
        for idx, (prompt, w_h) in enumerate(prompts):
            for i in range(5):
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    _,
                ) = pipe.encode_prompt(
                    prompt,
                    prompt,
                    device='cuda',
                )

                gc.collect()
                torch.cuda.empty_cache()

                print(prompt)
                width, height = w_h
                image = pipe(
                    prompt_embeds=prompt_embeds.to('cuda'),
                    pooled_prompt_embeds=pooled_prompt_embeds.to('cuda'),
                    width=width,
                    height=height,
                    num_inference_steps=20,
                    generator_seed=12345,
                    id_images=[subject_1, subject_2],
                    pulid_skip_timesteps=i+1,
                ).images[0]
                image.save(f'{INFERENCE_PATH}/{idx}_pulid_test_skip_{i+1}.jpg',
                            format='JPEG', subsampling=0, quality=95)


            for i in range(6):
                gc.collect()
                torch.cuda.empty_cache()

                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    _,
                ) = pipe.encode_prompt(
                    prompt,
                    prompt,
                    device='cuda',
                )
                gc.collect()
                torch.cuda.empty_cache()

                print(prompt)
                width, height = w_h
                image = pipe(
                    prompt_embeds=prompt_embeds.to('cuda'),
                    pooled_prompt_embeds=pooled_prompt_embeds.to('cuda'),
                    width=width,
                    height=height,
                    num_inference_steps=20,
                    generator_seed=12345,
                    id_images=[subject_1, subject_2],
                    pulid_skip_timesteps=2,
                    pulid_skip_end_timesteps=20 - i,
                ).images[0]
                image.save(f'{INFERENCE_PATH}/{idx}_pulid_test_skip_end_{20 - i}.jpg',
                            format='JPEG', subsampling=0, quality=95)

