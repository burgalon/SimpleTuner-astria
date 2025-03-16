import copy
import os
import sys

from pathlib import Path

# one level app + /astria
sys.path.append("astria")

import imagehash
from PIL import Image

from infer import *

from test_infer import BASE_PROMPT, run_images, JsonObj

# Do not send to server?
if 'DEBUG' not in os.environ:
    os.environ['DEBUG'] = 'test'

pipe = InferPipeline()

VIDEO_PARAMS = JsonObj(**{
    "id": "test-prompt-id",
    "text": "woman holding flowers",
    "tune_id": None,
    "num_images": 1,
    "tunes": [],
    "model_type": None,
})

def name():
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

test_name_invocation_count = {}

def test_img2video():
    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            '18046006-0.jpg' # 'pexels-damla-selen-demir-429137893-18166547.jpg'
    ).absolute())
    # prompt.text="Woman looking out into the ocean, she turns around to smile at the camera"
    prompt.video = True
    prompt.frames = 81
    prompt.text = 'A woman looking at the camera, camera moves around, her hair blowing in the wind'
    prompt.tunes=[]
    run_images(prompt)

def test_img2video_pipeline_swap():
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()
    run_images(BASE_PROMPT)

    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            '18046006-0.jpg' # 'pexels-damla-selen-demir-429137893-18166547.jpg'
    ).absolute())
    # prompt.text="Woman looking out into the ocean, she turns around to smile at the camera"
    prompt.video = True
    prompt.frames = 17
    prompt.text = 'A woman looking at the camera, camera moves around, her hair blowing in the wind'
    prompt.tunes=[]
    run_images(prompt)

    run_images(BASE_PROMPT)

def test_img2video_beach():
    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            'pexels-damla-selen-demir-429137893-18166547.jpg'
    ).absolute())
    # prompt.text="Woman looking out into the ocean, she turns around to smile at the camera"
    prompt.video = True
    prompt.frames = 81
    prompt.text = 'A woman is standing at a serene beach, she turns to smile at the camera as the wind blows through her hair'
    prompt.tunes=[]
    run_images(prompt)

def test_img2video_cat():
    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            'cattutu.png'
    ).absolute())
    # prompt.text="Woman looking out into the ocean, she turns around to smile at the camera"
    prompt.video = True
    prompt.frames = 81
    prompt.text = 'The cat in the tutu is dancing and does a twirl'
    prompt.tunes=[]
    run_images(prompt)

def test_img2video_cat_480p():
    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            'cattutu.png'
    ).absolute())
    # prompt.text="Woman looking out into the ocean, she turns around to smile at the camera"
    prompt.video = True
    prompt.frames = 81
    prompt.text = 'The cat in the tutu is dancing and does a twirl'
    prompt.tunes=[]
    run_images(prompt)