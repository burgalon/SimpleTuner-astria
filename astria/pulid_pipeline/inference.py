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

from optimum.quanto import freeze, quantize, qint8, qint4

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':
    from pipeline import FluxPipelineWithPulID
    from pulid_ext import PuLID

    base_model = "/data/models/1504944-flux1"
    location = "/home/user/storage/hf_cache"

    pipe = FluxPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        cache_dir=location,
    ).to('cuda')

    pipe(
        prompt='test',
        num_inference_steps=1,
    )

    pulid_model = PuLID(local_dir='/data/cache/pulid')

    pipe = FluxPipelineWithPulID(
        scheduler=pipe.scheduler,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer_2=pipe.tokenizer_2,
        vae=pipe.vae,
        transformer=pipe.transformer,
        pulid=pulid_model,
    ).to('cuda')

    input_image = '/app/astria_tests/fixtures/19477436-before-inpaint-0.jpg'
    input_image = 'test/margarethamilton.jpg'
    prompts = [
        (
            "A portrait photograph of a woman in a modern office.",
            (1024, 1024),
            input_image,
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

    # pipe.vae.to('cuda')
    # pipe._execution_device = torch.device('cuda')

    with torch.no_grad(), torch.inference_mode():
        for idx, (prompt, w_h, id_img_path) in enumerate(prompts):
            img = Image.open(id_img_path)
            pulid_embed, _ = pulid_model.get_id_embedding(img)
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

            images = []
            for scale in pulid_scales_to_test:
                generator = torch.Generator().manual_seed(12345)
                set_seed(12345)
                print(prompt)
                width, height = w_h
                image = pipe(
                    prompt_embeds=prompt_embeds.to('cuda'),
                    pooled_prompt_embeds=pooled_prompt_embeds.to('cuda'),
                    width=width,
                    height=height,
                    num_inference_steps=5,
                    generator=generator,
                    id_image_embeddings=pulid_embed,
                    id_image_scale=scale,
                ).images[0]
                image.save(f'{INFERENCE_PATH}/{idx}_pulid_test_{str(scale)}.jpg',
                           format='JPEG', subsampling=0, quality=95)
                images.append(add_text_to_image(image, 'cfg ' + str(scale)))

            fn = f'{INFERENCE_PATH}/{idx}_pulid_grid.jpg'
            make_image_grid(images, 3, 2).save(fn,
                                               format='JPEG', subsampling=0, quality=95)
            print('Saved to', fn)
