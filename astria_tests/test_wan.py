import copy
import os
import sys

from pathlib import Path

# one level app + /astria
sys.path.append("astria")

import imagehash
from PIL import Image

from infer import *

from test_infer import run_images, JsonObj

# Do not send to server?
if 'DEBUG' not in os.environ:
    os.environ['DEBUG'] = 'test'

pipe = InferPipeline()

VIDEO_PARAMS = JsonObj(**{
    "id": 12345,
    "name": "wan i2v test",
    "title": "wan i2v",
    "branch": "",
    "token": "",
    "task": "video",
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
    prompt.text = 'A woman looking at the camera, camera moves around, her hair blowing in the wind'
    prompt.tunes=[]
    run_images(prompt)