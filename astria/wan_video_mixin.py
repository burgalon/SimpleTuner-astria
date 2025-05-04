import gc
import os
import re
import tempfile
import torch

from pathlib import Path

from PIL import Image
from diffsynth import ModelManager, save_video
from huggingface_hub import snapshot_download
from wanvideo import WanVideoPipeline

from astria_utils import CACHE_DIR, MODELS_DIR, FileLock, JsonObj, run

if os.environ.get('MOCK_SERVER'):
    from astria_mock_server import send_to_server
else:
    from astria_send_to_server import send_to_server

WAN_I2V_LOCAL_LOCATION_480P = f"{CACHE_DIR}/Wan-AI/Wan2.1-I2V-14B-480P"
WAN_I2V_LOCAL_LOCATION_720P = f"{CACHE_DIR}/Wan-AI/Wan2.1-I2V-14B-720P"
WAN_VIDEO_MODEL_DETAILS = {
    "480p": {
        "name": "Wan-AI/Wan2.1-I2V-14B-480P",
        "location": WAN_I2V_LOCAL_LOCATION_480P,
        "resolution": (720, 720),
    },
    "720p": {
        "name": "Wan-AI/Wan2.1-I2V-14B-720P",
        "location": WAN_I2V_LOCAL_LOCATION_720P,
        "resolution": (1024, 1024),
    },
}
WAN_I2V_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


generate_ffmpeg_params = lambda width, height, watermark_path = None: [
    "-s", f"{width}x{height}",
    "-y",
    *(
        [
            "-i", watermark_path,
            "-filter_complex", f"[1:v]scale=-1:{int(height * 0.05)}[wm];[0:v][wm]overlay=x=(main_w-overlay_w-30):y=(main_h-overlay_h-30)"
        ]
        if watermark_path else []
    ),
    "-preset", "slow",
    "-pixel_format", "rgb24",
    "-map", "0:v",
    "-c:v", "libx265",
    # "-c:a", "aac",
    # "-shortest",
    "-pix_fmt", "yuv420p",
    "-crf", "28",
    "-colorspace", "bt709",
    "-tag:v", "hvc1",
    "-preset", "slow",
]


class WanVideoMixin:
    wan_i2v_pipe = None
    video_model_loaded = None
    wan_i2v_loras = None
    diffsynth_model_manager = None

    def init_wan_i2v_pipe(self, prompt, model_details):
        """Initialize the WAN video pipeline."""
        self.resolution = model_details["resolution"]
        lora_references = self.load_references_wan(prompt)
        lora_reference_key = [(lr["id"], lr["scale"]) for lr in lora_references]
        if lora_reference_key != []:
            print("Loading wan loras:", str(lora_reference_key))
        if (
            not self.wan_i2v_pipe or
            self.video_model_loaded != model_details["name"] or
            self.wan_i2v_loras != lora_reference_key
        ):
            # Ensure
            self.video_model_loaded = model_details["name"]
            location = model_details["location"]
            if not Path(location).exists():
                snapshot_download(
                    model_details["name"],
                    local_dir=location,
                )

            # Set to "cpu" for enable_vram_management below if needed
            model_manager = ModelManager(device="cuda")
            model_manager.load_models(
                [f"{location}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
                torch_dtype=torch.float32, # Image Encoder is loaded with float32
            )
            model_manager.load_models(
                [
                    [
                        f"{location}/diffusion_pytorch_model-00001-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00002-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00003-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00004-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00005-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00006-of-00007.safetensors",
                        f"{location}/diffusion_pytorch_model-00007-of-00007.safetensors",
                    ],
                    f"{location}/models_t5_umt5-xxl-enc-bf16.pth",
                    f"{location}/Wan2.1_VAE.pth",
                ],
                torch_dtype=torch.bfloat16, # Keep this in bfloat16... storing the weights as quant currently degrades the model
            )
            for lora_reference in lora_references:
                model_manager.load_lora(
                    lora_reference["location"],
                    lora_alpha=lora_reference["scale"],
                )
            self.wan_i2v_pipe = WanVideoPipeline.from_model_manager(
                model_manager,
                torch_dtype=torch.bfloat16,
                device="cuda",
            )

            # This is slow. Don't use it unless needed.
            # self.wan_i2v_pipe.enable_vram_management(num_persistent_param_in_dit=None)
            self.diffsynth_model_manager = model_manager

            # LoRAs that are loaded are stored in the tuple format (id, scaling).
            self.wan_i2v_loras = lora_reference_key

    def reset_wan_i2v(self, gc_collect=False):
        if self.wan_i2v_pipe is not None:
            self.video_model_loaded = None
            self.wan_i2v_pipe = None
            self.diffsynth_model_manager = None
            self.wan_i2v_loras = None

            if gc_collect:
                gc.collect()
                torch.cuda.empty_cache()

    def load_references_wan(self, prompt):
        lora_references = []
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

                if token == str(tune.id):
                    lora_references.append({
                        'id': str(tune.id),
                        'scale': float(scale),
                        'location': lora_fn,
                    })

                # Clean the match from the prompt
                prompt.text = prompt.text.replace(f"<{type}:{token}:{scale}>", "")
        return lora_references

    def infer_wan_i2v(self, prompt, tune: JsonObj, input_images = None):
        if input_images is None:
            input_images = [prompt.input_image]

        model_details = WAN_VIDEO_MODEL_DETAILS[prompt.video_model or "720p"]
        assert model_details is not None, f"unknown video model {prompt.video_model}"

        self.init_wan_i2v_pipe(prompt, model_details)

        negative_prompt = (
            prompt.negative_prompt
            if prompt.negative_prompt is not None
            else WAN_I2V_NEGATIVE_PROMPT
        )
        video_bytes_list = []
        for i, input_image in enumerate(input_images):
            prompt.input_image = input_image
            # Reset w,h to allow get_controlnet_hint to resize to same aspect ratio
            prompt.w = prompt.h = None
            _, _, _, _, _, input_image, _ = self.get_controlnet_hint(prompt)
            width, height = input_image.size
            print(f"Running WAN I2V on image {i} with size {width}x{height} prompt={prompt.video_prompt or prompt.text}")
            video = self.wan_i2v_pipe(
                prompt=prompt.video_prompt or prompt.text,
                negative_prompt=negative_prompt,
                input_image=input_image,
                num_inference_steps=prompt.steps or 30,
                seed=(prompt.seed or 42) + i,
                tiled=False,
                tea_cache_l1_thresh=0.15,
                tea_cache_model_id=model_details["name"].split("/")[1],
                num_frames=prompt.frames, # 5s default max length, or 81 frames
                width=width,
                height=height,
                slg_layers=[9],
                cfg_start=0.1,
                cfg_end=0.7,
            )

            # Create a temporary file in /dev/shm, keeping the video in RAM.
            with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".mp4", delete=True) as tmp:
                temp_path = tmp.name

                # Save your video to the temporary file.
                save_video(
                    video,
                    temp_path,
                    fps=prompt.fps,
                    ffmpeg_params=generate_ffmpeg_params(width, height),
                )

                # Read the contents into memory.
                with open(temp_path, "rb") as f:
                    video_bytes = f.read()

            video_bytes_list.append(input_image)
            video_bytes_list.append(video_bytes)

        if os.environ.get('DEBUG'):
            for i_video, video_b in enumerate(video_bytes_list):
                if not isinstance(video_b, Image.Image):
                    with open(f"{MODELS_DIR}/{prompt.id}-{i_video}.mp4", "wb") as f:
                        f.write(video_b)
                else:
                    video_b.save(f"{MODELS_DIR}/{prompt.id}-{i_video}.jpg")
        else:
            content_types = ["image/jpg", "video/mp4"] * len(video_bytes_list)
            send_to_server(video_bytes_list, prompt.id, content_types)

        return video_bytes
