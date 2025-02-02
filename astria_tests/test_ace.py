import copy
import json

import sys
sys.path.append("astria")

from infer import *
from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, FluxFillPipeline, FLUX_LORA_SHOE, \
    FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_CARTOON, FLUX_LORA_WOMAN_2, FLUX_LORA_DRESS, FLUX_LORA_COAT, FLUX_LORA_PANTS
from pathlib import Path

FLUX_ACE_TUNE_PORTRAIT = JsonObj(**{
    "id": -1,
    "name": "man",
    "title": "",
    "branch": "flux1",
    "model_type": "faceid",
    "face_swap_images": [
        'https://m.media-amazon.com/images/M/MV5BMTQ5NTUzNDE5OV5BMl5BanBnXkFtZTgwMjAwOTE1MDE@._V1_.jpg',
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

def test_fill_ace_portrait():
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        seed=42,
    )
    prompt.text=f"Maintain the facial features. A man is wearing a neat police uniform and sporting a badge. he is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo"
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_ace_portrait_cfg():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        fill_real_cfg=3.5,
        seed=42,
    )
    prompt.text=f"Maintain the facial features. A man is wearing a neat police uniform and sporting a badge. he is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo"
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_ace_portrait_cfg_slg():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        fill_real_cfg=3.5,
        fill_slg=json.dumps([[8, 12], [4, 7, 12]]),
        seed=42,
    )
    prompt.text=f"Maintain the facial features. A man is wearing a neat police uniform and sporting a badge. he is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo"
    prompt.tunes = [FLUX_ACE_TUNE_PORTRAIT]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_ace_subject():
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        seed=42,
    )
    prompt.text=f"Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table."
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_ace_subject_cfg():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        fill_real_cfg=3.5,
        seed=42,
    )
    prompt.text=f"Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table."
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_ace_subject_cfg_slg():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        ace_plus=True,
        fill_real_cfg=3.5,
        fill_slg=json.dumps([[8, 12], [4, 7, 12]]),
        seed=42,
    )
    prompt.text=f"Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table."
    prompt.tunes = [FLUX_ACE_TUNE_LOGO]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)