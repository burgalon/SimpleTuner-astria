import copy
import json

import sys
sys.path.append("astria")

from infer import *
from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, FluxFillPipeline, FLUX_LORA_SHOE, \
    FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_CARTOON, FLUX_LORA_WOMAN_2, FLUX_LORA_DRESS, FLUX_LORA_COAT, FLUX_LORA_PANTS
from pathlib import Path
from PIL import Image

FLUX_ACE_TUNE_PORTRAIT = JsonObj(**{
    "id": -1,
    "name": "woman",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        'https://sdbooth2-production.s3.amazonaws.com/l7pp1gzy1lthev4u1fiwvdt5l9tg',
    ],
})


FLUX_ACE_TUNE_PORTRAIT2 = JsonObj(**{
    "id": -1,
    "name": "woman",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        str(
            (Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / '19477436-before-inpaint-0.jpg').absolute()
        ),
    ],
})

FLUX_ACE_TUNE_PORTRAIT_ELON = JsonObj(**{
    "id": -1,
    "name": "man",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        "https://sdbooth2-production.s3.amazonaws.com/mj4x6ilylxma17ghadyu46emnuci",
    ],
})

FLUX_ACE_TUNE_FURNITURE = JsonObj(**{
    "id": -1,
    "name": "logo",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        str(
            (Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'furniture_reference.png').absolute()
        ),
    ],
})

FLUX_ACE_TUNE_LOGO = JsonObj(**{
    "id": -1,
    "name": "logo",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        'https://ichef.bbci.co.uk/images/ic/1920x1080/p03t1sm8.jpg',
    ],
})

FLUX_ACE_TUNE_BIKE = JsonObj(**{
    "id": -1,
    "name": "bike",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        'https://sdbooth2-production.s3.amazonaws.com/r3e0ehflknjlubkxgtfysgaf7jgj',
    ],
})


def test_fill_ace_portrait():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.text = ("Maintain the facial features. A woman is wearing a neat police uniform and "
                   "sporting a badge. She is smiling with a friendly and confident demeanor. "
                   "The background is blurred, featuring a cartoon logo")
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

    run_images(BASE_PROMPT)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_fill_ace_inpaint_portrait():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.cfg_scale = 50
    prompt.text = ("The man is facing the camera and is serious")
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT_ELON]
    prompt.input_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'portrait_inpaint_image.jpg'
    )
    prompt.mask_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'portrait_inpaint_mask.png',
    ).convert("L")
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_inpaint_portrait_cfg_slg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.fill_slg = json.dumps([[8, 12], [4, 7, 12]])
    prompt.text = ("The man is facing the camera and is serious")
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT2]
    prompt.input_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'portrait_inpaint_image.jpg'
    )
    prompt.mask_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'portrait_inpaint_mask.png',
    ).convert("L")
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_inpaint_furniture():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.cfg_scale = 50
    prompt.text = ("The furniture has a TV on it in the living room")
    prompt.tunes = [FLUX_ACE_TUNE_FURNITURE]
    prompt.input_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'furniture_inpaint_image.jpg'
    )
    prompt.mask_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'furniture_mask.png',
    ).convert("L")
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_inpaint_furniture_cfg_slg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.cfg_scale = 20
    prompt.fill_real_cfg = 2.8
    prompt.fill_slg = json.dumps([[8, 12], [4, 7, 12]])
    prompt.text = ("The furniture has a TV on it in the living room")
    prompt.tunes = [FLUX_ACE_TUNE_FURNITURE]
    prompt.input_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'furniture_inpaint_image.jpg'
    )
    prompt.mask_image = Image.open(
        Path(__file__).resolve().parent.parent / 'astria_tests' / 'fixtures' / 'furniture_mask.png',
    ).convert("L")
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_portrait_cfg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.seed = 42
    prompt.text = ("Maintain the facial features. A woman is wearing a neat police uniform and "
                   "sporting a badge. She is smiling with a friendly and confident demeanor. "
                   "The background is blurred, featuring a cartoon logo")
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_portrait_cfg_slg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.fill_slg = json.dumps([[8, 12], [4, 7, 12]])
    prompt.seed = 42
    prompt.text = ("Maintain the facial features. A woman is wearing a neat police uniform and "
                   "sporting a badge. She is smiling with a friendly and confident demeanor. "
                   "The background is blurred, featuring a cartoon logo")
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.text = ("Display the logo in a minimalist style printed in white on a matte black ceramic "
                   "coffee mug, alongside a steaming cup of coffee on a cozy cafe table.")
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject_cfg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.seed = 42
    prompt.text = ("Display the logo in a minimalist style printed in white on a matte black ceramic "
                   "coffee mug, alongside a steaming cup of coffee on a cozy cafe table.")
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject_cfg_slg():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.fill_slg = json.dumps([[8, 12], [4, 7, 12]])
    prompt.seed = 42
    prompt.text = ("Display the logo in a minimalist style printed in white on a matte black ceramic "
                   "coffee mug, alongside a steaming cup of coffee on a cozy cafe table.")
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject_bike():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.seed = 42
    prompt.h = 1024
    prompt.w = 1536
    prompt.text = ("A dramatic monochromatic studio portrait of a Triumph motorcycle, harsh rim lighting "
                   "creating sharp metallic highlights, deep shadows emphasizing mechanical curves and "
                   "textures, minimalist black backdrop, smoke elements adding atmosphere, shot with "
                   "Phase One medium format camera, precise lighting ratios, ultra-sharp details, high "
                   "contrast processing")
    prompt.tunes = [FLUX_ACE_TUNE_BIKE]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject_cfg_bike():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.seed = 42
    prompt.h = 1024
    prompt.w = 1536
    prompt.text = ("A dramatic monochromatic studio portrait of a Triumph motorcycle, harsh rim lighting "
                   "creating sharp metallic highlights, deep shadows emphasizing mechanical curves and "
                   "textures, minimalist black backdrop, smoke elements adding atmosphere, shot with "
                   "Phase One medium format camera, precise lighting ratios, ultra-sharp details, high "
                   "contrast processing")
    prompt.tunes = [FLUX_ACE_TUNE_BIKE]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"

def test_fill_ace_subject_cfg_slg_bike():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt.ace_plus = True
    prompt.cfg_scale = 30
    prompt.fill_real_cfg = 2.1
    prompt.fill_slg = json.dumps([[8, 12], [4, 7, 12]])
    prompt.seed = 42
    prompt.h = 1024
    prompt.w = 1536
    prompt.text = ("A dramatic monochromatic studio portrait of a Triumph motorcycle, harsh rim lighting "
                   "creating sharp metallic highlights, deep shadows emphasizing mechanical curves and "
                   "textures, minimalist black backdrop, smoke elements adding atmosphere, shot with "
                   "Phase One medium format camera, precise lighting ratios, ultra-sharp details, high "
                   "contrast processing")
    prompt.tunes = [FLUX_ACE_TUNE_BIKE]
    run_images(prompt)
    assert pipe.last_pipe == "ACE_plus"
