import copy
import sys
# one level app + /astria
sys.path.append("astria")

from infer import *
from astria_utils import JsonObj, MODELS_DIR

# Do not send to server?
if 'DEBUG' not in os.environ:
    os.environ['DEBUG'] = '1'

pipe = InferPipeline()

TUNE_FLUX = JsonObj(**{
    "id": 1504944,
    "name": None,
    "title": "Flux1.dev",
    "branch": "flux1",
    "token": "",
    "model_type": None,
})

FLUX_LORA = JsonObj(**{
    "id": 1533312,
    "name": "woman",
    "title": "Emma",
    "branch": "flux1",
    "token": str(1533312),
    "train_token": "ohwx",
    "model_type": "lora",
})

FLUX_FACEID = JsonObj(**{
    "id": 1533312,
    "name": "woman",
    "title": "Emma",
    "branch": "flux1",
    "token": str(1533312),
    "model_type": "faceid",
    "face_swap_images": [
        "https://sdbooth2-production.s3.amazonaws.com/l7pp1gzy1lthev4u1fiwvdt5l9tg",
    ]
})

FLUX_LORA_2 = JsonObj(**{
    "id": 1558021,
    "name": "man",
    "title": "Lance",
    "branch": "flux1",
    "token": str(1558021),
    "train_token": "ohwx",
    "model_type": "lora",
})

BASE_PROMPT = JsonObj(**{
    "id": "test-prompt-id",
    "text": "woman holding flowers",
    "tune_id": TUNE_FLUX.id,
    "num_images": 1,
    "tunes": [],
})

IMG_POSE = "https://sdbooth2-production.s3.amazonaws.com/hrzevyfi6cjj2o64c7xogyve1y13"

def name():
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

def run_images(prompt):
    prompt.id = name()
    tune = JsonObj(**TUNE_FLUX.__dict__, prompts=[prompt])
    images = pipe.infer(tune)
    # for i, image in enumerate(images):
    #     image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
    return images

# Test that loras do not leak across test by having a test of before/after lora
def test_txt2img_before():
    run_images(BASE_PROMPT)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_txt2img_lora():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_txt2img_after():
    run_images(BASE_PROMPT)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_faceid():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<{FLUX_FACEID.model_type}:{FLUX_FACEID.id}:1> woman holding flowers"
    prompt.tunes=[FLUX_FACEID]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipelineWithPulID)

def test_bad_faceid():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    FLUX_FACEID.face_swap_images = [
        'https://sdbooth2-production.s3.amazonaws.com/a93ocfwgzocdrmq1q4wizwajnhvm'
    ]
    prompt.text=f"<{FLUX_FACEID.model_type}:{FLUX_FACEID.id}:1> woman holding flowers"
    prompt.tunes=[FLUX_FACEID]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)


def test_superresolution():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
    )
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_hiresfix():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        hires_fix=True,
        super_resolution=True,
    )
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_img2img():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__), input_image=IMG_POSE)
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)

def test_controlnet_txt2img():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        controlnet='pose',
    )
    prompt.controlnet_txt2img = True
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --control_guidance_end 0.35"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetPipeline)

def test_controlnet_img2img():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        controlnet='pose',
    )
    prompt.controlnet_txt2img = False
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --control_guidance_end 0.35"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    prompt.denoising_strength=0.9
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetImg2ImgPipeline)


def test_inpainting_background_normal():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
    )
    prompt.text=f"lush forest --mask_prompt foreground --mask_invert --mask_dilate 0.5%"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_background_product():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # necklace
        # input_image='https://sdbooth2-production.s3.amazonaws.com/pwe6bcgo9ykbnt6tya91z3omoog4',
        # earings
        # input_image='https://sdbooth2-production.s3.amazonaws.com/u77j61zzd4gbqdsvkhmd9p1cz0d1',
        # dogs
        input_image='https://sdbooth2-production.s3.amazonaws.com/a93ocfwgzocdrmq1q4wizwajnhvm',
        denoising_strength=1,
        # cfg_scale=3,
    )
    # prompt.num_images = 8
    prompt.w = prompt.h = None
    prompt.text=f"studio shot, on a rock, surrounded by tropical vegetation, visible moisture in the air, water drops, over a blurry background, drama, high contrast image, teal and orange photo filter --mask_prompt background --mask_dilate 0 --mask_blur 0 --mask_inc_brightness 10"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_foreground():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman with black blouse --mask_prompt foreground --mask_dilate 1%"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_controlnet_foreground():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
        controlnet='pose',
    )
    prompt.controlnet_txt2img = True
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman with black blouse --control_guidance_end 0.35 --mask_prompt foreground --mask_dilate 1%"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetInpaintPipeline)


def test_film_grain():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__), film_grain=True)
    run_images(prompt)

def test_clut():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__), color_grading='Film Velvia')
    run_images(prompt)
