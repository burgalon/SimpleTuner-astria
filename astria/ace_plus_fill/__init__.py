import random
from typing import Optional, Union, List
from collections import OrderedDict
from pathlib import Path

import torch, os
from diffusers import FluxFillPipeline
from scepter.modules.utils.config import Config
# from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
# from scepter.modules.utils.logger import get_logger
from transformers import T5TokenizerFast
from .utils import ACEPlusImageProcessor, scale_long_edge_and_pad, random_crop_pil
from huggingface_hub import snapshot_download
from PIL import Image

HF_REPO = "ali-vilab/ACE_Plus"
LOCAL_SUBDIR = "ace_plus"
SUBJECT_LORA_NAME = "comfyui_subject_lora16.safetensors"
PORTRAIT_LORA_NAME = "comfyui_portrait_lora64.safetensors"
LOCAL_EDITING_LORA_NAME = "comfyui_local_lora16.safetensors"

ACE_NEGATIVE_PORTRAIT_PROMPT = "a blurry picture of several sea turtles with hot air balloons in the background, a chicken drumstick in the foreground on a blue, pink, and green plate"
ACE_NEGATIVE_SUBJECT_PROMPT = "a low quality jpeg image of an old playground, hot air balloons in the background, a chicken drumstick in the foreground on a blue, pink, and green plate"

UNCOND_IMAGE_LOC = Path(__file__).resolve().parent / 'static' / 'uncond_ace.jpg'


class ACEPlusDiffuserInference():
    def __init__(self):
        self.input = {}
        self.pipe = None
        self.pipe_location = None
        self.device = None

    def load_default(self, cfg):
        if cfg is not None:
            self.input_cfg = {k.lower(): v for k, v in cfg.INPUT.items()}
            self.input = {
                k.lower(): dict(v).get('DEFAULT', None)
                if isinstance(v, (dict, OrderedDict, Config))
                else v for k, v in cfg.INPUT.items()
            }
            self.output = {k.lower(): v for k, v in cfg.OUTPUT.items()}

    @classmethod
    def init_from_pipe(cls, pipe, pipe_location, device, ace_location, ace_model):
        self = cls()
        self.device = device
        original_tokenizer_2 = pipe.tokenizer_2

        # Lower this to 4096 or 3172 if you start to OOM
        self.max_seq_len = 3072 # cfg.get("MAX_SEQ_LEN", 4096)
        self.image_processor = ACEPlusImageProcessor(max_seq_len=self.max_seq_len)

        # This does nothing if it already exists
        os.makedirs(os.path.join(ace_location, LOCAL_SUBDIR), exist_ok=True)
        snapshot_download(HF_REPO, local_dir=os.path.join(ace_location, LOCAL_SUBDIR),
            allow_patterns=["*.safetensors"])

        self.pipe_location = pipe_location
        tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(pipe_location, "tokenizer_2"),
            additional_special_tokens=["{image}"])
        
        self.pipe = pipe
        self.pipe.register_modules(tokenizer_2=tokenizer_2)

        if ace_model == 'subject':
            self.pipe.load_lora_weights(
                f"{ace_location}/{LOCAL_SUBDIR}/{ace_model}/{SUBJECT_LORA_NAME}",
                adapter_name="ace_lora_adapter",
                low_cpu_mem_usage=True,
            )
        if ace_model == 'portrait':
            self.pipe.load_lora_weights(
                f"{ace_location}/{LOCAL_SUBDIR}/{ace_model}/{PORTRAIT_LORA_NAME}",
                adapter_name="ace_lora_adapter",
                low_cpu_mem_usage=True,
            )
        if ace_model == 'local_editing':
            self.pipe.load_lora_weights(
                f"{ace_location}/{LOCAL_SUBDIR}/{ace_model}/{LOCAL_EDITING_LORA_NAME}",
                adapter_name="ace_lora_adapter",
                low_cpu_mem_usage=True,
            )

        def post_process_fn(img, slice_w, out_w, out_h):
            return self.image_processor.postprocess(img, slice_w, out_w, out_h)

        def unload_function():
            self.pipe.delete_adapters(["ace_lora_adapter"])
            self.pipe.register_modules(tokenizer_2=original_tokenizer_2)

        return self, post_process_fn, unload_function

        # self.pipe.tokenizer_2 = tokenizer_2
        # self.load_default(cfg.DEFAULT_PARAS)

    def prepare_input(self,
                      image,
                      mask,
                      batch_size=1,
                      dtype = torch.bfloat16,
                      num_images_per_prompt=1,
                      height=512,
                      width=512,
                      generator=None):
        num_channels_latents = self.pipe.vae.config.latent_channels
        # import pdb;pdb.set_trace()
        mask, masked_image_latents = self.pipe.prepare_mask_latents(
            mask.unsqueeze(0),
            image.unsqueeze(0).to(self.device, dtype = dtype),
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            dtype,
            torch.cuda.current_device(), # we.device_id,
            generator,
        )
        # import pdb;pdb.set_trace()
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
        return masked_image_latents

    def prepare_inputs(
        self,
        prompt_text: Union[str, List[str]],
        repainting_scale: float = 0.0,
        reference_image: Optional[Image.Image] = None,
        edit_image: Optional[Image.Image] = None,
        edit_mask: Optional[Image.Image] = None,
        seed=42,
    ) -> torch.Tensor:
        # TODO is this needed?
        reference_image = scale_long_edge_and_pad(reference_image)

        if isinstance(prompt_text, str):
            prompt = [prompt_text]
        if all(inp is None for inp in (reference_image, edit_image, edit_mask)):
            raise ValueError('reference_image, edit_image, and/or edit_mask required')
        reference_image_sz = reference_image.size

        seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        uncond_img = Image.open(UNCOND_IMAGE_LOC)
        uncond_img = random_crop_pil(uncond_img, reference_image_sz)
        uncond_img, mask, out_h, out_w, slice_w = self.image_processor.preprocess(
            uncond_img, edit_image, edit_mask, repainting_scale = repainting_scale)
        h, w = uncond_img.shape[1:]
        generator = torch.Generator("cpu").manual_seed(seed)
        masked_image_latents_uncond = self.prepare_input(uncond_img, mask,
            batch_size=len(prompt), height=h, width=w, generator=generator)

        image, mask, out_h, out_w, slice_w = self.image_processor.preprocess(
            reference_image, edit_image, edit_mask, repainting_scale = repainting_scale)
        h, w = image.shape[1:]
        generator = torch.Generator("cpu").manual_seed(seed + 1)
        masked_image_latents = self.prepare_input(image, mask,
            batch_size=len(prompt), height=h, width=w, generator=generator)

        return masked_image_latents, masked_image_latents_uncond, h, w, slice_w, out_w, out_h

    @torch.no_grad()
    def __call__(
        self,
        reference_image=None,
        edit_image=None,
        edit_mask=None,
        prompt='',
        task=None,
        output_height=1024,
        output_width=1024,
        sampler='flow_euler',
        sample_steps=28,
        guide_scale=50,
        lora_path=None,
        seed=-1,
        tar_index=0,
        align=0,
        repainting_scale=0,
        **kwargs,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        image, mask, out_h, out_w, slice_w = self.image_processor.preprocess(
            reference_image, edit_image, edit_mask, repainting_scale = repainting_scale)
        h, w = image.shape[1:]
        generator = torch.Generator("cpu").manual_seed(seed)
        masked_image_latents = self.prepare_input(image, mask,
            batch_size=len(prompt) , height=h, width=w, generator = generator)

        if lora_path is not None:
            with FS.get_from(lora_path) as local_path:
                self.pipe.load_lora_weights(local_path)

        image = self.pipe(
            prompt=prompt,
            masked_image_latents=masked_image_latents,
            height=h,
            width=w,
            guidance_scale=guide_scale,
            num_inference_steps=sample_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        if lora_path is not None:
            self.pipe.unload_lora_weights()
        return self.image_processor.postprocess(image, slice_w, out_w, out_h), seed
