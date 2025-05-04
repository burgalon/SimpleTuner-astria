import copy
import os
import sys

from pathlib import Path

# one level app + /astria
sys.path.append("astria")

import requests
from PIL import Image

from infer import *

from astria.astria_utils import MODELS_DIR
from test_infer import BASE_PROMPT, run_images, JsonObj

# Do not send to server?
if 'DEBUG' not in os.environ:
    os.environ['DEBUG'] = 'test'

pipe = InferPipeline()

SQUISH_LORA_URL = "https://huggingface.co/Remade-AI/Squish/resolve/main/squish_18.safetensors"

VIDEO_PARAMS = JsonObj(**{
    "id": "test-prompt-id",
    "text": "woman holding flowers",
    "tune_id": None,
    "num_images": 1,
    "tunes": [],
    "model_type": None,
})

VIDEO_TUNE = JsonObj(**{
    "id": "squish_18",
    "name": "squish",
    "title": "Squish",
    "branch": "wan",
    "token": "squish",
    "train_token": "sq41sh",
    "model_type": "lora",
})

def ensure_file_exists(url: str, target_path: Path, chunk_size: int = 8192) -> None:
    """
    Ensure that the file at target_path exists. If not, download it from the given URL.

    Args:
        url (str): URL to download the file from.
        target_path (Path): Local path where the file should be saved.
        chunk_size (int, optional): Size of each chunk to download. Defaults to 8192.

    Raises:
        Exception: If the HTTP request fails.
    """
    if target_path.exists():
        return  # File already exists, nothing to do.

    # Ensure the target directory exists.
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the file in streaming mode.
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks.
                    f.write(chunk)
    else:
        raise Exception(f"Failed to download file from {url}: HTTP {response.status_code}")


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

def test_img2video_woman2():
    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            'wan_woman_fail.jpg'
    ).absolute())
    prompt.video = True
    prompt.frames = 81
    prompt.text = 'woman smiling, looking at the camera, bright eyes'
    prompt.tunes=[]
    run_images(prompt)

def test_img2video_pipeline_swap():
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
    prompt.frames = 2
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
    prompt.denoising_strength = 0.0
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
    prompt.denoising_strength = 0.0
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
    prompt.video_model = "480p"
    prompt.frames = 81
    prompt.text = 'The cat in the tutu is dancing and does a twirl'
    prompt.tunes=[]
    run_images(prompt)


def test_img2video_cat_480p_lora():
    target_file = Path(f"{MODELS_DIR}/squish_18.safetensors")
    ensure_file_exists(SQUISH_LORA_URL, target_file)

    prompt = JsonObj(
        **copy.copy(VIDEO_PARAMS.__dict__),
    )
    prompt.input_image = str((
        Path(__file__).resolve().parent.parent /
            'astria_tests' /
            'fixtures' /
            'cattutu.png'
    ).absolute())
    prompt.video = True
    prompt.video_model = "480p"
    prompt.frames = 81
    prompt.text = f'<lora:{VIDEO_TUNE.id}:1> In the video, a cat in a tutu is presented. The cat in a tutu is held in a person’s hands. The person then presses on the cat in a tutu, causing a sq41sh squish effect. The person keeps pressing down on the cat in a tutu, further showing the sq41sh squish effect.'
    prompt.tunes=[VIDEO_TUNE]
    run_images(prompt)
