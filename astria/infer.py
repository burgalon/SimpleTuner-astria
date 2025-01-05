import argparse
import os
import random
import re
import shlex
import sys
import time
import traceback

from filelock import FileLock
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlNetPipeline, FluxControlNetModel, \
    FluxInpaintPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline
from torchvision import transforms

from pulid_pipeline.pipeline import FluxPipelineWithPulID
from pulid_pipeline.pulid_ext import PuLID

from ragdiffusion import RAG_FluxPipeline, openai_gpt4o_get_regions

from add_clut import add_clut
from add_grain import add_grain
from astria_utils import run, MODELS_DIR, download_model_from_server, JsonObj, device, \
    StaleDeploymentException, FLUX_INPAINT_MODEL_ID, CACHE_DIR, \
    HUMAN_CLASS_NAMES, check_refresh

if os.environ.get('MOCK_SERVER'):
    from astria_mock_server import report_infer_job_failure, request_infer_job_from_server, request_tune_job_from_server, send_to_server
else:
    from astria_server import report_infer_job_failure, request_infer_job_from_server, request_tune_job_from_server
    from astria_send_to_server import send_to_server

from birefnet.BiRefNet_node import BiRefNet_node
from controlnet_constants import CONTROLNETS_DICT, CONTROL_MODES
from hinter_helper import get_detector
from image_utils import load_image, load_images
from pipeline_flux_differential_img2img import FluxDifferentialImg2ImgPipeline
from runpod_utils import kill_pod
from sig_listener import TerminateException, is_terminated, set_current_infer_tune, set_current_train_tune
from super_resolution_helper import load_sr, upscale_sr
from train import train
from inpaint_face_mixin import InpaintFaceMixin
from vton_mixin import VtonMixin, VTON_CATEGORIES
from watermark_helper import add_watermark

PIL2TENSOR = transforms.Compose([transforms.PILToTensor()])
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU_MEMORY_GB={GPU_MEMORY_GB:.0f}")

def parse_args(prompt: JsonObj):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_prompt", type=str, default=None)
    parser.add_argument("--mask_negative", type=str, nargs="+", default=[])
    parser.add_argument("--mask_index", type=int, default=None)
    parser.add_argument("--mask_dilate", type=str, default=0)
    parser.add_argument("--mask_blur", type=int, default=0)
    parser.add_argument("--mask_inc_brightness", type=int, default=None)
    parser.add_argument("--mask_invert", action='store_true', default=False)
    parser.add_argument("--disable_restore_mask_area", action='store_true', default=None)
    parser.add_argument("--face_inpaint_denoising", type=float, default=None)
    parser.add_argument("--hires_denoising_strength", type=float, default=None)
    parser.add_argument("--fill", action='store_true', default=False)
    parser.add_argument("--outpaint", choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', 'top-center', 'bottom-center', 'left-center', 'right-center'], default=None)
    parser.add_argument("--outpaint_height", type=int, default=None)
    parser.add_argument("--outpaint_width", type=int, default=None)
    parser.add_argument("--outpaint_prompt", type=str, default=None)
    parser.add_argument("--restore_mask", action='store_true', default=False)
    parser.add_argument("--face_swap_indexes", type=int, nargs="+")
    # parser.add_argument("--denoising_strength", type=float, default=prompt.denoising_strength)
    parser.add_argument("--limit_frames", type=int, help="Max frames to process", default=None)
    parser.add_argument("--max_interval", type=int, help="Maximum number of frames between keyframes", default=12)
    parser.add_argument("--min_interval", type=int, help="Minimum number of frames between keyframes", default=4)
    parser.add_argument("--threshold", type=int, help="Minimum movement change to create a new keyframe", default=10)
    parser.add_argument("--control_types", type=str, nargs="+", choices=CONTROLNETS_DICT['flux1'].keys(), default=["pose", "depth", "lineart", "tile"])
    parser.add_argument("--control_guidance_start", type=float, default=None)
    parser.add_argument("--control_guidance_end", type=float, default=None)
    parser.add_argument("--conditioning_scales", type=float, nargs="+", default=None)
    parser.add_argument("--depth_threshold", type=int, default=0, help="Mask scene by depth 0-255")
    parser.add_argument("--fast_keyframes_generation", action='store_true', help="Smaller grid => faster keyframes generation but less consistent")
    parser.add_argument("--lineart_grid", action='store_true', help="Use grid instead of lineart hints for lineart controlnet")
    parser.add_argument("--loras", type=int, nargs="+", help="LoRa model id from Civit")
    parser.add_argument("--lora_weights", type=float, nargs="+")
    parser.add_argument("--resolution_factor", type=float, default=1, help="Resolution factor to use for generate")
    parser.add_argument("--upscale_factor", type=int, default=None, help="Upscale factor to use for generate")
    parser.add_argument("--tiled_upscale", action='store_true', help="Tiled upscaling", default=prompt.tiled_upscale or os.environ.get('TILED_UPSCALE'))
    parser.add_argument("--only_upscale", action='store_true', help="Only upscale without txt2img or img2img", default=prompt.only_upscale or os.environ.get('ONLY_UPSCALE'))
    parser.add_argument("--input_faceid", type=float, help="Use input_image for faceid. Defines the scale of the input_image for faceid", default=None, const=1.0, nargs='?')
    parser.add_argument("--faceid_portrait", help="Use faceid portrait", action='store_true', default=False)
    parser.add_argument("--fix_bindi", help="Inpaint dot on the forehead", action='store_true', default=False)
    parser.add_argument("--vton_cfg_scale", help="VTON cfg_scale", type=float, default=None)
    parser.add_argument("--vton_hires", help="VTON Hi resolution", action='store_true', default=False)
    parser.add_argument(
        "--use_regional",
        help="Uses RAG diffusion with gpt4o prompt enhancement", action='store_true',
        default=getattr(prompt, 'use_regional', False),  # Will be False or None
    )
    parser.add_argument(
        "--regional_hb_replace",
        type=int,
        default=prompt.regional_hb_replace
            if getattr(prompt, 'regional_hb_replace', None) is not None
            else 2,
        help="HB replace",
    )
    parser.add_argument(
        "--regional_sr_delta",
        type=float,
        default=prompt.regional_sr_delta
            if getattr(prompt, 'regional_sr_delta', None) is not None
            else 1.0,
        help="SR delta",
    )
    parser.add_argument('text', nargs='*', help='Text to be processed')


    # Other inference
    parser.add_argument('--controlnet_txt2img', action='store_true', help="Use controlnet txt2img instead of img2img", default=prompt.controlnet_txt2img or False)


    try:
        args, unknown = parser.parse_known_args(shlex.split(prompt.text.replace("'", "\\'")))
        if args.hires_denoising_strength is not None:
            args.hires_denoising_strength = min(1.0, max(0.1, args.hires_denoising_strength))
    except (SystemExit, Exception, TypeError) as e:
        traceback.print_exc()
        return
    # assign args to prompt
    for k, v in vars(args).items():
        setattr(prompt, k, v)
    prompt.text = " ".join(args.text+unknown)
    if prompt.cfg_scale:
        prompt.cfg_scale = float(prompt.cfg_scale)
    if prompt.only_upscale:
        print(f"T#{prompt.tune_id} P#{prompt.id} Only upscaling - resetting other attributes including controlnet")
        prompt.super_resolution = False
        prompt.face_swap_images = []
        prompt.hires_fix = False
        prompt.inpaint_faces = False
        prompt.controlnet = None

def get_pipe_key_for_lora(pipe):
    return 'fill' if isinstance(pipe, FluxFillPipeline) else 'pipe'

class InferPipeline(InpaintFaceMixin, VtonMixin):
    reference_pattern = r'<(lora|faceid):([^>:]+):([\d\.]+)>'
    reference_pattern_re = re.compile(reference_pattern)

    def __init__(self):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        InpaintFaceMixin.__init__(self)
        self.reset()

    def reset(self):
        self.last_pipe = None
        self.model_path = None
        self.pipe = None
        self.img2img = None
        self.inpaint = None
        self.current_lora_weights_map = {}
        self.resolution = None

        self.reset_controlnet()

    def reset_controlnet(self):
        self.sr_model = None
        self.control = None
        self.control_type = None
        self.controlnet_txt2img = None
        self.controlnet_img2img = None
        self.controlnet_inpaint_txt2img = None
        if 'fill' in self.current_lora_weights_map:
            del self.current_lora_weights_map['fill']
        self.fill = None
        self.pulid_model = None
        self.pulid_pipe = None
        self.rag_diffusion_pipe = None
        torch.cuda.empty_cache()


    def warmup(self):
        start_time = time.time()
        self.init_pipe(MODELS_DIR + "/1504944-flux1")
        print(f"Initialized pipeline in {time.time() - start_time:.2f}s")
        start_time = time.time()
        self.init_inpaint(JsonObj(fill=True))
        print(f"Initialized inpaint in {time.time() - start_time:.2f}s")

    def unload_lora_weights(self, pipe):
        pipe_key = get_pipe_key_for_lora(pipe)
        if pipe_key == 'fill':
            self.fill.unload_lora_weights()
        else:
            self.pipe.unload_lora_weights()
        self.current_lora_weights_map[pipe_key] = {'names': [], 'scales': []}

    def load_references(self, prompt: JsonObj, pipe):
        names = []
        scales = []
        lora_fns = []
        setattr(prompt, '_prompt_with_lora_ids', prompt.text)
        pipe_key = get_pipe_key_for_lora(pipe)
        if pipe_key not in self.current_lora_weights_map:
            self.current_lora_weights_map[pipe_key] = {'names': [], 'scales': []}
        current_lora_weights = self.current_lora_weights_map[pipe_key]

        all_match_groups = re.findall(self.reference_pattern_re, prompt.text)
        for match_groups in all_match_groups:
            type, token, scale = match_groups
            if type == "lora":
                tune = next(iter([tune for tune in prompt.tunes if tune.token == token or str(tune.id) == token]), None)
                if not tune:
                    raise Exception(f"Token {token} not found in prompt {prompt.id} tokens={prompt.tunes}")
                if os.environ.get('LORA_FN'):
                    lora_fn = os.environ.get('LORA_FN')
                else:
                    lora_fn = f"{MODELS_DIR}/{tune.id}.safetensors"
                with FileLock(f"{lora_fn}.lock", timeout=60):
                    if not os.path.exists(lora_fn):
                        run(['aws', 's3', 'cp', f"s3://sdbooth2-production/models/{tune.id}.safetensors", lora_fn])
                    else:
                        # touch for access time so that it doesn't get cleaned up
                        os.utime(lora_fn)

                # This is hack fix for bug in diffusers doe due to the new ` _maybe_expand_lora_state_dict`
                # when first loading a small rank lora and then a higher rank lora
                # and so Civit models which atr represented by token strings are usually higher ranks, so we load first
                # until this is fixed in diffusers
                if token == str(tune.id):
                    names.append(str(tune.id))
                    scales.append(float(scale))
                    lora_fns.append(lora_fn)
                else:
                    # insert at the beginning
                    names.insert(0, str(tune.id))
                    scales.insert(0, float(scale))
                    lora_fns.insert(0, lora_fn)

                # Clean the match from the prompt
                prompt.text = prompt.text.replace(f"<{type}:{token}:{scale}>", "")
                prompt._prompt_with_lora_ids = prompt._prompt_with_lora_ids.replace(f"<{type}:{token}:{scale}>", f"{str(tune.id)}")

        if len(names) == 0:
            # No LoRA weights needed for this prompt
            if current_lora_weights.get('names', []):
                print(f"Unloading LoRA weights for pipe={pipe.__class__.__name__}")
                # Unload any previously loaded LoRA weights
                self.unload_lora_weights(pipe)
            return {}

        # Compare with current LoRA weights
        if names == current_lora_weights.get('names', []) and scales == current_lora_weights.get('scales', []):
            print("LoRA weights already loaded with the same scales")
        else:
            if current_lora_weights.get('names', []):
                self.unload_lora_weights(pipe)
            # Load new LoRA weights
            start_time = time.time()
            for name, lora_fn in zip(names, lora_fns):
                print(f"Loading LoRA weights {name} from {lora_fn}")
                pipe.load_lora_weights(lora_fn, adapter_name=name, low_cpu_mem_usage=True)
            print(f"Loaded LoRA weights {names} in {time.time() - start_time:.2f}s pipe={pipe.__class__.__name__}")
            current_lora_weights['names'] = names
            current_lora_weights['scales'] = scales
            self.current_lora_weights_map[pipe_key] = current_lora_weights
            if len(self.current_lora_weights_map.keys()) > 2:
                print(f"current_lora_weights_map={[k.__class__.__name__ for k in self.current_lora_weights_map.keys()]}")
                raise ValueError("current_lora_weights_map too large")

        # Set adapters
        pipe.set_adapters(names, adapter_weights=scales)

        print(f"set_adapters names={names} scales={scales}")

        if len(names) > 1:
            return {"scale": 1.0}
        return {"scale": scales[0]}

    def init_pipe(self, model_path):
        """Initialize both FluxPipeline and FluxImg2ImgPipeline."""
        if not self.pipe or self.model_path != model_path:
            # Initialize the FluxPipeline for text-to-image
            self.pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(device)
            self.model_path = model_path
            # TODO: Remove once this is merged to diffusers
            self.resolution = (1024, 1024)

    def init_pulid(self):
        if not self.pulid_pipe:
            os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
            self.pulid_model = PuLID(local_dir=f'{CACHE_DIR}/pulid')
            self.pulid_model.clip_vision_model.to(device)

            self.pulid_pipe = FluxPipelineWithPulID(
                scheduler=self.pipe.scheduler,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer_2=self.pipe.tokenizer_2,
                vae=self.pipe.vae,
                transformer=self.pipe.transformer,
                pulid=self.pulid_model,
            ).to(device)

    def get_pulid_embedding(self, tune: JsonObj):
        embedding_file = f"{MODELS_DIR}/{tune.id}_pulid_embedding.pt"
        lock_file = f"{embedding_file}.lock"

        with FileLock(lock_file, timeout=60):
            if os.path.exists(embedding_file) and not os.environ.get("FORCE_PULID"):
                if os.path.getsize(embedding_file) == 0:
                    raise RuntimeError(f"T#{tune.id} No embeddings can be calculated for this tune.")
                # Load the embedding and update access time
                print(f"T#{tune.id} Loading PulID embedding")
                pulid_embed = torch.load(embedding_file)
                os.utime(embedding_file)
                return pulid_embed

            try:
                # Calculate the embedding and save it
                start_time = time.time()
                pulid_embed, _ = self.pulid_model.get_id_embedding_for_images_list(load_images(tune.face_swap_images[:4]))
                print(f"T#{tune.id} Calculated PulID embedding in {time.time() - start_time:.2f}s")
                torch.save(pulid_embed, embedding_file)
                os.utime(embedding_file)
            except RuntimeError:
                # Create a zero-length file to indicate failure
                open(embedding_file, 'w').close()
                raise

        return pulid_embed

    def init_rag_diffusion(self):
        if not self.rag_diffusion_pipe:
            os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
            self.rag_diffusion_pipe = RAG_FluxPipeline(
                scheduler=self.pipe.scheduler,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer_2=self.pipe.tokenizer_2,
                vae=self.pipe.vae,
                transformer=self.pipe.transformer,
            ).to(device)

    def init_img2img(self):
        if not self.img2img:
            # Initialize the FluxImg2ImgPipeline for image-to-image
            self.img2img = FluxImg2ImgPipeline(
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            ).to(device)

    def init_control(self, tune: JsonObj, control_type):
        control_type = CONTROLNETS_DICT[tune.branch][control_type]
        if self.control_type != control_type:
            self.control_type = control_type
            self.control = FluxControlNetModel.from_pretrained(
                self.control_type,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(device)
            if self.controlnet_txt2img:
                self.controlnet_txt2img.controlnet = self.control
            if self.controlnet_img2img:
                self.controlnet_img2img.controlnet = self.control
            if self.controlnet_inpaint_txt2img:
                self.controlnet_inpaint_txt2img.controlnet = self.control

    def init_inpaint(self, prompt: JsonObj):
        if prompt.fill:
            if not self.fill:
                model_path = download_model_from_server(f'{FLUX_INPAINT_MODEL_ID}-flux1')
                self.fill = FluxFillPipeline(
                    transformer=FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.bfloat16),
                    scheduler=self.pipe.scheduler,
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                ).to("cuda")
            return self.fill
        else:
            if not self.inpaint:
                self.inpaint = FluxDifferentialImg2ImgPipeline(
                    transformer=self.pipe.transformer,
                    scheduler=self.pipe.scheduler,
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                ).to(device)
            return self.inpaint

    def init_controlnet_inpaint_txt2img(self, tune, control_type):
        self.init_control(tune, control_type)
        if not self.controlnet_inpaint_txt2img:
            self.controlnet_inpaint_txt2img = FluxControlNetInpaintPipeline(
                controlnet=self.control,
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            ).to(device)

    def init_controlnet_img2img(self, tune, control_type):
        self.init_control(tune, control_type)
        if not self.controlnet_img2img:
            self.controlnet_img2img = FluxControlNetImg2ImgPipeline(
                controlnet=self.control,
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            ).to(device)

    def init_controlnet_txt2img(self, tune, control_type):
        self.init_control(tune, control_type)
        if not self.controlnet_txt2img:
            self.controlnet_txt2img = FluxControlNetPipeline(
                controlnet=self.control,
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            ).to(device)


    def poll_infer(self):
        tune = JsonObj()
        try:
            tune = request_infer_job_from_server()
            if tune.id is None:
                return 0

            self.infer(tune)
        except StaleDeploymentException:
            print("Stale deployment. Shutting down")
            kill_pod()
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()
            if tune.prompts:
                for prompt in tune.prompts:
                    if prompt.trained_at is None:
                        report_infer_job_failure(prompt, traceback.format_exc())
            raise e
        return 1

    def get_controlnet_hint(self, prompt: JsonObj, orig_mask_image: Image.Image = None):
        assert prompt.input_image is not None
        try:
            orig_input_image = load_image(prompt.input_image)
        except Image.DecompressionBombError:
            print(f"T#{prompt.tune_id} P#{prompt.id} DecompressionBombError - skipping")
            prompt.controlnet = None
            prompt.input_image = None
            return None, None, None, None, None, None, None

        # To match controlnet hints
        # https://github.com/patrickvonplaten/controlnet_aux/blob/fbaf09fa847d914c85cf24647dab6b4cf1740eca/src/controlnet_aux/open_pose/util.py#L224
        # following Discussion https://discord.com/channels/@me/1086436595642671115/1111651021530333255
        # Bring back requested resolution
        if prompt.w and prompt.h:
            # Depth and inpainting specifically requires 32x32
            w, h = int(np.round(prompt.w / 32.0) * 32), int(np.round(prompt.h / 32.0) * 32)
        else:
            w, h = orig_input_image.size
            # if we're doing outpainting only, we want the output target to be always the same size
            # so esssentially we only want to downsample (resize down) the input image
            # moreover, when resizing we do not want to maintain the same total number of pixels but rather
            # make sure the largest dimension is the same as the target size
            if prompt.denoising_strength == 0 and prompt.outpaint:
                if w>prompt.outpaint_width or h>prompt.outpaint_height:
                    if float(w)/prompt.outpaint_width > float(h) / prompt.outpaint_height:
                        k = prompt.outpaint_width / w
                        print(f"T#{prompt.tune_id} P#{prompt.id} outpaint_width={prompt.outpaint_width} w={w} k={k:.2f}")
                    else:
                        k = prompt.outpaint_height / h
                        print(f"T#{prompt.tune_id} P#{prompt.id} outpaint_height={prompt.outpaint_height} h={h} k={k:.2f}")
                else:
                    k =1
                    print(f"T#{prompt.tune_id} P#{prompt.id} outpaint={prompt.outpaint_width}x{prompt.outpaint_height} input_image.size={orig_input_image.size} k=1")
            else:
                k = (float(self.resolution[0] * self.resolution[1]) / (w * h)) ** 0.5

            h *= k
            w *= k
            h = int(np.round(h / 32.0)) * 32
            w = int(np.round(w / 32.0)) * 32

        input_image = ImageOps.fit(orig_input_image, (w, h), method=Image.LANCZOS, bleed=0.0, centering=(0.5, 0.5))
        mask_image = ImageOps.fit(orig_mask_image, (w, h), method=Image.NEAREST, bleed=0.0, centering=(0.5, 0.5)) if orig_mask_image else None
        if os.environ.get('DEBUG'):
            input_image.save(f"{MODELS_DIR}/{prompt.id}-input.jpg")
            if mask_image:
                mask_image.save(f"{MODELS_DIR}/{prompt.id}-mask-input.jpg")
        # helps detect if image is already a pose image
        mean = np.mean(np.array(input_image))
        if prompt.controlnet_hint:
            controlnet_hint = load_image(prompt.controlnet_hint_image)
        elif mean < 15:
            controlnet_hint = input_image
            if not prompt.controlnet_txt2img:
                print(f"T#{prompt.tune_id} P#{prompt.id} controlnet input image is pre-processing but controlnet is not txt2img - setting controlnet_txt2img=True")
                prompt.controlnet_txt2img = True
        elif prompt.controlnet:
            start_time = time.time()
            prompt.controlnet_hint = controlnet_hint = get_detector(prompt.controlnet)(input_image)
            print(f"T#{prompt.tune_id} P#{prompt.id} get_controlnet_hint hinter {prompt.controlnet} took {time.time() - start_time:.2f}s - controlnet_hint.size={controlnet_hint.size} input_image={input_image.size} input_image.size={input_image.size}")
        else:
            controlnet_hint = None
            print(f"T#{prompt.tune_id} P#{prompt.id} controlnet_hint_image is not provided and controlnet is not set - skipping controlnet. input_image.size={input_image.size}")

        if os.environ.get('DEBUG') == 'controlnet':
            controlnet_hint.save(f"{MODELS_DIR}/{prompt.id}-{prompt.controlnet}.jpg")
            input_image.save(f"{MODELS_DIR}/{prompt.id}-input.jpg")

        input_image_tensor = PIL2TENSOR(input_image)
        input_image_tensor = input_image_tensor / 255 * 2 - 1

        return input_image_tensor, controlnet_hint, w, h, orig_input_image, input_image, mask_image

    def outpaint(self, images, prompt, kwargs):
        out = []
        pipe = self.init_inpaint(JsonObj(fill=True))
        print(f"T#{prompt.tune_id} P#{prompt.id} outpaint {prompt.outpaint} {prompt.outpaint_height}x{prompt.outpaint_width} text={prompt.outpaint_prompt} from {images[0].size}")
        h = prompt.outpaint_height
        w = prompt.outpaint_width
        if not h or not w:
            return images
        if w*h > 2048*2048:
            k = (2048*2048 / (w*h)) ** 0.5
            new_h = int(np.round(h * k))
            new_w = int(np.round(w * k))
            print(f"T#{prompt.tune_id} P#{prompt.id} outpaint {w}x{h} is too large, resizing to {new_w}x{new_h}")
            w, h = new_w, new_h

        if prompt.outpaint_prompt is None:
            prompt.outpaint_prompt = prompt.text
        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt.outpaint_prompt,
            prompt.outpaint_prompt,
            max_sequence_length=prompt.max_sequence_length or 512,
            device=device,
        )

        for i_image, input_image in enumerate(images):
            new_image = Image.new("RGB", (w, h), (255, 255, 255))
            mask_image = Image.new("L", (w, h), 255)
            black_rectangle = Image.new("L", (input_image.width, input_image.height), 0)
            if prompt.outpaint == 'top-left':
                new_image.paste(input_image, (0, 0))
                mask_image.paste(black_rectangle, (0, 0))
            elif prompt.outpaint == 'top-right':
                new_image.paste(input_image, (w - input_image.width, 0))
                mask_image.paste(black_rectangle, (w - input_image.width, 0))
            elif prompt.outpaint == 'top-center':
                new_image.paste(input_image, ((w - input_image.width) // 2, 0))
                mask_image.paste(black_rectangle, ((w - input_image.width) // 2, 0))
            elif prompt.outpaint == 'bottom-left':
                new_image.paste(input_image, (0, h - input_image.height))
                mask_image.paste(black_rectangle, (0, h - input_image.height))
            elif prompt.outpaint == 'bottom-right':
                new_image.paste(input_image, (w - input_image.width, h - input_image.height))
                mask_image.paste(black_rectangle, (w - input_image.width, h - input_image.height))
            elif prompt.outpaint == 'bottom-center':
                new_image.paste(input_image, ((w - input_image.width) // 2, h - input_image.height))
                mask_image.paste(black_rectangle, ((w - input_image.width) // 2, h - input_image.height))
            elif prompt.outpaint == 'left-center':
                new_image.paste(input_image, (0, (h - input_image.height) // 2))
                mask_image.paste(black_rectangle, (0, (h - input_image.height) // 2))
            elif prompt.outpaint == 'right-center':
                new_image.paste(input_image, (w - input_image.width, (h - input_image.height) // 2))
                mask_image.paste(black_rectangle, (w - input_image.width, (h - input_image.height) // 2))
            elif prompt.outpaint == 'center':
                new_image.paste(input_image, ((w - input_image.width) // 2, (h - input_image.height) // 2))
                mask_image.paste(black_rectangle, ((w - input_image.width) // 2, (h - input_image.height) // 2))
            else:
                print(f"T#{prompt.tune_id} P#{prompt.id} Invalid outpaint {prompt.outpaint}")
                out.append(input_image)
                continue
            if os.environ.get('DEBUG'):
                new_image.save(f"{MODELS_DIR}/{prompt.id}-{i_image}-outpaint.jpg")
                mask_image.save(f"{MODELS_DIR}/{prompt.id}-{i_image}-mask-outpaint.jpg")

            image = pipe(
                guidance_scale=30,
                height=h,
                width=w,
                num_inference_steps=prompt.steps or 28,
                generator=torch.Generator(device="cuda").manual_seed((prompt.seed or 42) + i_image),
                image=new_image,
                mask_image=mask_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            ).images[0]
            out.append(image)
        return out

    def apply_hires_fix(self, images, prompt, kwargs):
        self.init_img2img()
        for i_image in range(len(images)):
            if is_terminated():
                raise TerminateException("terminated")

            strength = prompt.hires_denoising_strength or 0.25

            image = images[i_image]
            if image.width % 32 != 0 or image.height % 32 != 0:
                # resize to nearest 32x32
                image = ImageOps.fit(
                    image,
                    (int(np.round(image.width / 32.0) * 32), int(np.round(image.height / 32.0) * 32)),
                    method=Image.LANCZOS,
                    bleed=0.0,
                    centering=(0.5, 0.5)
                )

            print(f"T#{prompt.tune_id} P#{prompt.id} hires_fix {i_image=} {images[i_image].width}x{images[i_image].height} {strength=}")
            image = self.img2img(
                image=image,
                strength=strength,
                guidance_scale=float(prompt.cfg_scale or 3.5),
                height=image.height,
                width=image.width,
                num_inference_steps=28,
                max_sequence_length=prompt.max_sequence_length or 512,
                generator=torch.Generator(device="cuda").manual_seed((prompt.seed or 42) + i_image),
                prompt_embeds=kwargs['prompt_embeds'],
                pooled_prompt_embeds=kwargs['pooled_prompt_embeds'],
            ).images[0]
            images[i_image] = image

        return images

    def infer(self, tune: JsonObj):
        set_current_infer_tune(tune)
        model_path = download_model_from_server(f"{tune.id}-{tune.branch}")
        self.init_pipe(model_path)

        images = None
        for i_prompt, prompt in enumerate(tune.prompts):
            start_time = time.time()
            images = self.infer_prompt(prompt, tune)
            print(f"T={tune.id} {i_prompt}/{len(tune.prompts)} P={prompt.id} U={prompt.user_id} https://www.astria.ai/admin/prompts/{prompt.id} {(time.time() - start_time):.2f} seconds")
        set_current_infer_tune(None)
        return images

    def infer_mask(self, prompt):
        print(f"T#{prompt.tune_id} P#{prompt.id} Infer mask {prompt.mask_prompt=}")
        if prompt.mask_prompt == 'background':
            prompt.mask_prompt = 'foreground'
            prompt.mask_invert = True

        birefnet = BiRefNet_node()
        image = load_image(prompt.input_image)
        alpha_tensor : torch.Tensor = birefnet.matting(image, 'cuda')
        alpha = alpha_tensor.squeeze().cpu().numpy()
        mask = Image.fromarray((alpha * 255).astype(np.uint8))

        if prompt.mask_invert:
            mask = ImageOps.invert(mask)

        if prompt.mask_dilate:
            dilation = int(float(prompt.mask_dilate[:-1])/100 * self.resolution[0] if prompt.mask_dilate.endswith('%') else prompt.mask_dilate)
            print(f"{dilation=}")
            if dilation > 0:
                kernel = np.ones((dilation, dilation), np.uint8)
                mask = cv2.dilate(np.array(mask), kernel, iterations=1)
                mask = Image.fromarray(mask)
            else:
                kernel = np.ones((-dilation, -dilation), np.uint8)
                mask = cv2.erode(np.array(mask), kernel, iterations=1)
                mask = Image.fromarray(mask)

        if prompt.mask_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(prompt.mask_blur))

        if prompt.mask_inc_brightness:
            # increase mask brightness so that black are now more gray
            mask = mask.point(lambda p: p + prompt.mask_inc_brightness)

        if os.environ.get('DEBUG'):
            mask.save(f"{MODELS_DIR}/{prompt.id}-mask.jpg")
        return mask

    def upscale(self, images, prompt):
        if not self.sr_model:
            self.sr_model = load_sr(f"/data/cache/4x_NMKD-Siax_200k.pth")
        for i_image, image in enumerate(images):
            if os.environ.get('DEBUG'):
                image.save(f"{MODELS_DIR}/{prompt.id}-{i_image}-before-esrgan.jpg")

            if os.environ.get('UPSCALE_FACTOR'):
                upscale_factor = int(os.environ.get('UPSCALE_FACTOR'))
            elif prompt.upscale_factor:
                upscale_factor = prompt.upscale_factor
            else:
                upscale_factor = 4 if image.width * image.height < 512 * 512 else 2

            images[i_image] = upscale_sr(self.sr_model, image, upscale_factor)
        return images

    def infer_prompt(self, prompt, tune: JsonObj):
        parse_args(prompt)
        if prompt.w and prompt.h:
            prompt.w, prompt.h = int(np.round(prompt.w / 32.0) * 32), int(np.round(prompt.h / 32.0) * 32)

        kwargs = {}

        if prompt.mask_prompt and prompt.input_image and not prompt.mask_image:
            prompt.mask_image = self.infer_mask(prompt)

        input_image_tensor, controlnet_hint, w, h, orig_input_image, input_image, mask_image = None, None, None, None, None, None, None
        use_regional =  getattr(prompt, 'use_regional', False)

        # Note that the below condition is IMPORTANT and need to be modified cautiously
        # See test_vton_img2img_strength0
        if any([tune.model_type == 'faceid' and tune.name in HUMAN_CLASS_NAMES for tune in prompt.tunes]):
            if input_image:
                raise Exception("Cannot have both faceid and input_image")
            if use_regional:
                raise Exception("Can not use face ID with --use_regional")
            for match_groups in re.findall(self.reference_pattern_re, prompt.text):
                type, token, scale = match_groups
                if type == "faceid":
                    tune = next(iter([tune for tune in prompt.tunes if tune.token == token or str(tune.id) == token]), None)
                    if not tune:
                        raise Exception(f"Token {token} not found in prompt {prompt.id} tokens={prompt.tunes}")
                    if not tune.face_swap_images:
                        raise Exception(f"Token {token} has no face_swap_images")

                    # Clean the match from the prompt
                    # also needs to be cleaned for VTON
                    prompt.text = prompt.text.replace(f"<{type}:{token}:{scale}>", "")

                    if tune.name in VTON_CATEGORIES:
                        pipe = self.pipe
                        continue
                    if tune.name not in HUMAN_CLASS_NAMES:
                        raise Exception(f"Token {token} is not a human")
                    self.init_pulid()
                    try:
                        kwargs['id_image_embeddings'] = self.get_pulid_embedding(tune)
                        kwargs['id_image_scale'] = float(scale)
                        pipe = self.pulid_pipe
                    except RuntimeError:
                        print(f"T#{prompt.tune_id} P#{prompt.id} Face not detected - skipping")
                        pipe = self.pipe
                        continue
                    print(f"T#{prompt.tune_id} P#{prompt.id} faceid={token} scale={scale}")
                    break
        elif prompt.input_image:
            if prompt.mask_image:
                orig_mask_image = load_image(prompt.mask_image, "L")
            else:
                orig_mask_image = None
            input_image_tensor, controlnet_hint, w, h, orig_input_image, input_image, mask_image = self.get_controlnet_hint(prompt, orig_mask_image)
            prompt.w = prompt.w or w
            prompt.h = prompt.h or h

            if prompt.controlnet:
                kwargs['control_image'] = controlnet_hint
                kwargs['control_mode'] = CONTROL_MODES[prompt.controlnet]
                kwargs['controlnet_conditioning_scale'] = float(prompt.denoising_strength if prompt.denoising_strength != None else 0.8)

                if prompt.control_guidance_start:
                    kwargs['control_guidance_start'] = prompt.control_guidance_start
                if prompt.control_guidance_end:
                    kwargs['control_guidance_end'] = prompt.control_guidance_end

                if mask_image:
                    self.init_controlnet_inpaint_txt2img(tune, prompt.controlnet)
                    pipe = self.controlnet_inpaint_txt2img
                    kwargs['mask_image'] = mask_image
                    kwargs['image'] = input_image
                else:
                    if prompt.controlnet_txt2img:
                        self.init_controlnet_txt2img(tune, prompt.controlnet)
                        pipe = self.controlnet_txt2img
                    else:
                        self.init_controlnet_img2img(tune, prompt.controlnet)
                        pipe = self.controlnet_img2img
                        kwargs['strength'] = float(prompt.denoising_strength or 0.8)
                        kwargs['image'] = input_image
            else:
                kwargs['image'] = input_image
                if mask_image:
                    pipe = self.init_inpaint(prompt)
                    if isinstance(pipe, FluxDifferentialImg2ImgPipeline):
                        print("Inverting mask for differential diffusion")
                        mask_image = ImageOps.invert(mask_image)
                    kwargs['mask_image'] = mask_image
                else:
                    self.init_img2img()
                    pipe = self.img2img

                if isinstance(pipe, FluxFillPipeline):
                    if not prompt.cfg_scale or prompt.cfg_scale < 7:
                        prompt.cfg_scale = 30
                else:
                    kwargs['strength'] = float(prompt.denoising_strength if prompt.denoising_strength != None else 0.8)
        elif use_regional:
            self.init_rag_diffusion()
            pipe = self.rag_diffusion_pipe
        else:
            pipe = self.pipe

        # For tests
        self.last_pipe = pipe
        images = []
        num_images = int(os.environ.get('NUM_IMAGES', prompt.num_images))
        joint_attention_kwargs = self.load_references(prompt)
        prompt.text = prompt.text.strip(" ,").strip(" ").strip('"')

        # load_references mutates prompt.text, so this needs to be down here.
        if use_regional:
            HB_replace =  getattr(prompt, 'regional_hb_replace', 2)
            SR_delta =  getattr(prompt, 'regional_sr_delta', 1.0)

            regions = openai_gpt4o_get_regions(prompt._prompt_with_lora_ids)
            print(f"T#{prompt.tune_id} P#{prompt.id} regions={regions}")

            # Now that we have the regions, we need to assign LoRAs to each
            # region, if relevant.
            HB_prompt_list = regions["HB_prompt_list"]
            SR_prompt = regions["SR_prompt"]
            lora_regional_scaling = []
            sr_prompts = [p.strip() for p in SR_prompt.split("BREAK")]
            for sr_prompt in sr_prompts:
                scaling_values = {
                    lora_id: 0.0
                    for lora_id in self.current_lora_weights.get('names', [])
                }
                for lora_id, lora_scale in zip(
                    self.current_lora_weights.get('names', []),
                    self.current_lora_weights.get('scales', []),
                ):
                    if lora_id in sr_prompt:
                        scaling_values[lora_id] = lora_scale
                lora_regional_scaling.append(scaling_values)

            # Clean the prompts of any LoRA IDs that might have been embedded.
            HB_prompt_list_cleaned = []
            for hb_prompt in HB_prompt_list:
                for lora_id in self.current_lora_weights.get('names', []):
                    hb_prompt = hb_prompt.replace(lora_id, "")
                HB_prompt_list_cleaned.append(hb_prompt)

            for lora_id in self.current_lora_weights.get('names', []):
                SR_prompt = SR_prompt.replace(lora_id, "")

            HB_replace = HB_replace
            HB_m_offset_list = regions["HB_m_offset_list"]
            HB_n_offset_list = regions["HB_n_offset_list"]
            HB_m_scale_list = regions["HB_m_scale_list"]
            HB_n_scale_list = regions["HB_n_scale_list"]
            SR_delta = SR_delta
            SR_hw_split_ratio = regions["SR_hw_split_ratio"]

            kwargs = {
                **kwargs,
                **dict(
                    SR_delta=SR_delta,
                    SR_hw_split_ratio=SR_hw_split_ratio,
                    SR_prompt=SR_prompt,
                    HB_prompt_list=HB_prompt_list_cleaned,
                    HB_m_offset_list=HB_m_offset_list,
                    HB_n_offset_list=HB_n_offset_list,
                    HB_m_scale_list=HB_m_scale_list,
                    HB_n_scale_list=HB_n_scale_list,
                    HB_replace=HB_replace,
                    seed=prompt.seed or 42,
                    lora_regional_scaling=lora_regional_scaling,
                ),
            }

        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt.text,
            prompt.text,
            max_sequence_length=prompt.max_sequence_length or 512,
            device=device,
        )
        kwargs['prompt_embeds'] = prompt_embeds
        kwargs['pooled_prompt_embeds'] = pooled_prompt_embeds

        print(f"T#{prompt.tune_id} P#{prompt.id} pipe={pipe.__class__.__name__} {prompt.text=} loras={self.current_lora_weights_map[get_pipe_key_for_lora(pipe)]}")
        for i_image in range(num_images):
            if 'image' in kwargs is not None and 'strength' in kwargs and kwargs['strength'] == 0:
                print(f"T#{prompt.tune_id} P#{prompt.id} Skipping image {i_image} because strength=0. Probably just VTON or outpaint only?")
                images.append(kwargs['image'])
                continue
            image = pipe(
                guidance_scale=float(prompt.cfg_scale if prompt.cfg_scale is not None else 3.5),
                height=prompt.h or 1024,
                width=prompt.w or 1024,
                num_inference_steps=prompt.steps or 28,
                generator=torch.Generator(device="cuda").manual_seed((prompt.seed or 42) + i_image),
                joint_attention_kwargs=joint_attention_kwargs,
                **kwargs,
            ).images[0]
            if is_terminated():
                raise TerminateException("terminated")
            images.append(image)

        if prompt.outpaint:
            images = self.outpaint(images, prompt, kwargs)

        images = self.vton(images, prompt)
        if prompt.super_resolution or os.environ.get('SUPER_RESOLUTION'):
            images = self.upscale(images, prompt)

        if prompt.hires_fix or os.environ.get('HIRES_FIX'):
            if os.environ.get('DEBUG'):
                for i_image, image in enumerate(images):
                    image.save(f"{MODELS_DIR}/{prompt.id}-{i_image}-before-hires.jpg")
            images = self.apply_hires_fix(images, prompt, kwargs)

        if prompt.inpaint_faces or os.environ.get('INPAINT_FACES'):
            images = self.inpaint_faces(images, prompt, tune)

        if prompt.color_grading and prompt.color_grading != 'null' and not os.environ.get('DISABLE_COLOR_GRADING'):
            print(f"T#{prompt.tune_id} P#{prompt.id} color_grading={prompt.color_grading}")
            images = [add_clut(image, prompt.color_grading) for image in images]
        if prompt.film_grain or os.environ.get('FILM_GRAIN'):
            print(f"T#{prompt.tune_id} P#{prompt.id} film_grain={prompt.film_grain}")
            images = [add_grain(image) for image in images]

        if prompt.base_pack and prompt.base_pack.watermark:
            images = add_watermark(images, prompt.base_pack.watermark)

        # if it's inpainting, we need to resize the mask, and then paste back the original image
        if prompt.restore_mask:
            print(f"T#{prompt.tune_id} P#{prompt.id} Restoring background")

            if not mask_image:
                mask_image = self.infer_mask(prompt).resize(input_image.size)

            for i_image, image in enumerate(images):
                # in case of super-resolution, input_image and mask_image are different resolution, so we need to resize
                image = Image.composite(image, input_image.resize(image.size, Image.LANCZOS), mask_image.resize(image.size, Image.LANCZOS))
                images[i_image] = image


        if os.environ.get('DEBUG'):
            for i_image, image in enumerate(images):
                image.save(f"{MODELS_DIR}/{prompt.id}-{i_image}.jpg")
        else:
            send_to_server(images, prompt.id)
        prompt.trained_at = True
        return images

def main():
    download_model_from_server(f"1504944-flux1")
    pipeline = InferPipeline()

    def poll():
        if GPU_MEMORY_GB > 50:
            pipeline.warmup()
        i = 0
        max_sleeps = int(os.environ.get('MAX_SLEEPS', 90))
        while not is_terminated() and (i < max_sleeps or os.environ.get('DONT_STOP')):
            check_refresh()
            processed_jobs = pipeline.poll_infer()

            # Give a few chances for inference before starting training
            if not os.environ.get('DISABLE_TRAINING'):
                tune = request_tune_job_from_server()
                if tune.id:
                    if GPU_MEMORY_GB < 50 or (tune.args and 'preprocessing' in tune.args):
                        pipeline.reset()
                    else:
                        pipeline.reset_controlnet()
                    print(f"Training tune {tune.id}")
                    set_current_train_tune(tune)
                    train(tune)
                    set_current_train_tune(None)
                    processed_jobs = 1
                    pipeline.warmup()

            # Check both train + infer
            if processed_jobs == 0:
                i += 1
                time.sleep(2)
                print(f"{i}.", end="")
            else:
                i = 0
        print(f"Exiting poll after i={i}")

    import sys

    if len(sys.argv) == 1:
        print("No arguments provided. Assuming poll")
        sys.argv.append("poll")

    if 'stop' not in sys.argv and 'poll' in sys.argv:
        print("Polling started without stop. Setting DONT_STOP=1")
        os.environ['DONT_STOP'] = "1"

    for id in sys.argv[1:]:
        if id == 'stop':
            kill_pod()
            break
        if id == "poll":
            poll()
            continue
        tune = request_infer_job_from_server(id=id) if id.isnumeric() else request_infer_job_from_server(tune_id=id[1:])
        if not tune.id:
            print(f"No tune found for {id}")
            continue
        pipeline.infer(tune)

if __name__ == "__main__":
    main()
