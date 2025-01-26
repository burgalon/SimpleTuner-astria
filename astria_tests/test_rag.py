import copy
import json

import sys
sys.path.append("astria")

from infer import *
from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, RAG_FluxPipeline, FLUX_LORA_SHOE, \
    FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_CARTOON, FLUX_LORA_WOMAN_2
from pathlib import Path


def test_regional_normal():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )
    prompt.text = "a dog inbetween two vases full of flowers. the flowers on the left are white lillies, the flowers on the right are roses. the dog is a pembroke welsh corgi. above the corgi, there are balloons flying that say \"happy birthday\""
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

    run_images(JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    ))

"""
def test_regional_multi_run_multi_lora():
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.tunes=[FLUX_LORA]
    prompt.text = f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman inbetween two vases full of flowers. the flowers on the left are white lillies, the flowers on the right are roses. above the {FLUX_LORA.train_token} woman, there are balloons flying that say \"happy birthday\""
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man, real photograph portrait holding <lora:{FLUX_CARTOON.id}:1> {FLUX_CARTOON.train_token} sloth, white t-shirt, white background, professional headshot with cartoon sloth character. The top half of the image is the man's shoulder and face, which the bottom half of the image is the 2d illustration being held in his arms. Compositing cinematography example, cartoon overlaid on photograph"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_CARTOON]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)
"""

def test_regional_lora():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_lora.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )
    prompt.tunes=[FLUX_LORA]
    prompt.text = f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman inbetween two vases full of flowers. the flowers on the left are white lillies, the flowers on the right are roses. above the {FLUX_LORA.train_token} woman, there are balloons flying that say \"happy birthday\""
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_two_lora():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_two_lora.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_two_lora_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman and <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man all seated together on a rollercoaster cart with 2 seats. They are all thrilled as the rollercoaster races down a decline, their hair blown back by the wind."
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_three_lora():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_three_lora.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man, <lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man standing next to the couple"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_three_lora_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )

    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man, <lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman, and <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man all seated together on a rollercoaster cart with 3 seats. They are all thrilled as the rollercoaster races down a decline, their hair blown back by the wind."
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_four_lora_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        h=768,
        w=1280,
    )

    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man, <lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman, <lora:{FLUX_LORA_WOMAN_2.id}:1> {FLUX_LORA_WOMAN_2.train_token} woman, and <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man all seated together on a rollercoaster cart with 4 seats. They are all thrilled as the rollercoaster races down a decline, their hair blown back by the wind."
    prompt.tunes=[FLUX_LORA, FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_LORA_WOMAN_2]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_one_lora_famous_person():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers standing beside donald trump in the white house"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_two_lora_person_and_object():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_two_lora_person_and_object.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man with <lora:{FLUX_LORA_SHOE.id}:1> {FLUX_LORA_SHOE.train_token} shoe on and prominently visible, walking on a gravel path in a forest"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_LORA_SHOE]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

def test_regional_one_lora_person_and_cartoon_lora():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_one_lora_person_and_cartoon_lora.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man, real photograph portrait holding <lora:{FLUX_CARTOON.id}:1> {FLUX_CARTOON.train_token} sloth, white t-shirt, white background, professional headshot with cartoon sloth character. The top half of the image is the man's shoulder and face, which the bottom half of the image is the 2d illustration being held in his arms. Compositing cinematography example, cartoon overlaid on photograph"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_CARTOON]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, RAG_FluxPipeline)
