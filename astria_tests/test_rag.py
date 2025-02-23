import copy
import json

import sys
sys.path.append("astria")

from infer import *
from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, RAG_FluxPipeline, FLUX_LORA_SHOE, \
    FLUX_LORA_MAN_MARCO, FLUX_LORA_MAN, FLUX_CARTOON, FLUX_LORA_WOMAN_2, FLUX_LORA_DRESS, FLUX_LORA_COAT, FLUX_LORA_PANTS
from pathlib import Path


FLUX_LORA_GAME_UI = JsonObj(**{
    "id": 2163359,
    "name": "style",
    "title": "game ui style",
    "branch": "flux1",
    "token": str(2163359),
    "train_token": "ukj",
    "model_type": "lora",
    "face_swap_images": []
})
FLUX_LORA_GAME_UI_2 = JsonObj(**{
    "id": 2158702,
    "name": "style",
    "title": "game ui style",
    "branch": "flux1",
    "token": str(2158702),
    "train_token": "ukj",
    "model_type": "lora",
    "face_swap_images": []
})




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
    # assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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

def test_regional_three_lora_complex_violence_force_llm():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional="force_llm",
    )

    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man tied to metal interrogation chair, screaming in terror, wearing bloodstained mustard corduroy jacket and loose tie and <lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman in hotel uniform with white blouse and purple vest pressing against him, head on his shoulder, crying, and <lora:{FLUX_LORA_MAN_MARCO.id}:1> {FLUX_LORA_MAN_MARCO.train_token} man in denim sherpa jacket holding silenced pistol to the other man's head. In stark interrogation room with dramatic spotlight, teal and amber grading"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_LORA_MAN_MARCO, FLUX_LORA]
    run_images(prompt)
    assert pipe.last_pipe is None
   

def test_regional_two_lora_with_dress():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_two_lora_person_and_full_body.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman is posing indoors wearing a <lora:{FLUX_LORA_DRESS.id}:1> {FLUX_LORA_DRESS.train_token} dress"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_DRESS]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_two_lora_with_full_body_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman is posing indoors wearing a <lora:{FLUX_LORA_DRESS.id}:1> {FLUX_LORA_DRESS.train_token} dress"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_DRESS]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_two_lora_with_upper_body():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_two_lora_person_and_upper_body.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )

    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man is posing outdoors on a ski hill wearing a <lora:{FLUX_LORA_COAT.id}:1> {FLUX_LORA_COAT.train_token} coat"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_LORA_COAT]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_two_lora_with_upper_body_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
    )

    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man is posing outdoors on a ski hill wearing a <lora:{FLUX_LORA_COAT.id}:1> {FLUX_LORA_COAT.train_token} coat"
    prompt.tunes=[FLUX_LORA_MAN, FLUX_LORA_COAT]
    run_images(prompt)
    assert pipe.last_pipe is None

def test_regional_two_lora_with_lower_body():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_two_lora_person_and_lower_body.json'
    with open(json_pth, 'r') as f:
        regional_json = json.dumps(json.load(f))
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
    )

    prompt.text=f"full body photo of <lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman is a at a fancy luncheon wearing <lora:{FLUX_LORA_PANTS.id}:1> {FLUX_LORA_PANTS.train_token} pants"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_PANTS]
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

def test_regional_two_lora_women_no_premade_json():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        h=768,
        w=1280,
    )

    prompt.text=f"{FLUX_LORA_WOMAN_2.train_token} {FLUX_LORA_WOMAN_2.name} <lora:{FLUX_LORA_WOMAN_2.id}:1> and {FLUX_LORA.train_token} {FLUX_LORA.name} <lora:{FLUX_LORA.id}:1> side be side in a  business portrait photoshoot --multi"
    prompt.tunes=[FLUX_LORA, FLUX_LORA_WOMAN_2]
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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)

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
#     assert isinstance(pipe.last_pipe, RAG_FluxPipeline)


def test_regional_game_ui():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_ui.json'
    with open(json_pth, 'r') as f:
        regional_json = json.load(f)

    regional_json['SR_prompt'] = """
Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

Top banner, on a bright purple scroll, text says "READY, SETS, GO!". The words READY and GO! are in a gold color. The text SETS is in a white color. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

Box with text inside that says "TODAY ONLY!", purple background, gold rim. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

Text that says "GET x2 ON THESE SETS REWARDS". the "x2" text is huge compared to the other text. The text is all white except for "REWARDS", which is gold. At the bottom, there are three icons with the labels "Peonella", "Candy Kids", and "Alice In Peonland" beneath them. The icons are cute cartoons. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

An orange cartoon character rides a rocket ship with an outstretched hand. He smiles at the viewer. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

A large green button with the word "AWESOME!" on it in white text. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. BREAK
"""
    regional_json = json.dumps(regional_json)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
        w=1526,
        h=1024,
        regional_sr_delta=1.0,
        regional_hb_replace=3.0,
    )
    prompt.text=f"Opaque modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background."
    prompt.tunes=[]
    run_images(prompt)

def test_regional_game_ui_lora():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_ui.json'
    with open(json_pth, 'r') as f:
        regional_json = json.load(f)

    regional_json['SR_prompt'] = f"""
Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style.BREAK

Top banner, on a bright purple scroll, text says "READY, SETS, GO!". The words READY and GO! are in a gold color. The text SETS is in a white color. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

Box with text inside that says "TODAY ONLY!", purple background, gold rim. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

Text that says "GET x2 ON THESE SETS REWARDS". the "x2" text is huge compared to the other text. The text is all white except for "REWARDS", which is gold. At the bottom, there are three icons with the labels "Peonella", "Candy Kids", and "Alice In Peonland" beneath them. The icons are cute cartoons. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

An orange cartoon character rides a rocket ship with an outstretched hand. He smiles at the viewer. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

A large green button with the word "AWESOME!" on it in white text. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} {FLUX_LORA_GAME_UI_2.train_token} style. BREAK
"""
    regional_json = json.dumps(regional_json)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
        w=1526,
        h=1024,
        regional_sr_delta=1.0,
        regional_hb_replace=2.0,
    )
    prompt.text=f"<lora:{FLUX_LORA_GAME_UI_2.id}:1> {FLUX_LORA_GAME_UI_2.train_token} style. Opaque Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background."
    prompt.tunes=[FLUX_LORA_GAME_UI_2]
    run_images(prompt)

def test_regional_game_ui_lora_alt_text():
    regional_json = ''
    json_pth = Path(__file__).resolve().parent.parent / 'astria' / 'ragdiffusion' / 'prompts' / 'test_regional_ui.json'
    with open(json_pth, 'r') as f:
        regional_json = json.load(f)

    regional_json['SR_prompt'] = f"""
Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner.BREAK

Top banner, on a bright purple scroll, text says "GET SOME COINS!". The words GET and SOME are in a gold color. The text COINS is in a white color. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

Box with text inside that says "ENJOY NOW!", purple background, gold rim. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

Text that says "DAILY DOUBLE ON REWARDS". the "double" text is huge compared to the other text. The text is all white except for "REWARDS", which is gold. At the bottom, there are three icons with the labels "Chocolate", "Vanilla", and "Strawberry" beneath them. The icons are cute cartoons. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

An orange cartoon character rides a rocket ship with an outstretched hand. He smiles at the viewer. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

A large green button with the word "LET'S GO!" on it in white text. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK

A pile of gold coins with a blue cards flying around them. There are spirals on the cards. Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background. {FLUX_LORA_GAME_UI_2.id} monetization gaming banner. BREAK
"""
    regional_json = json.dumps(regional_json)
    with open('test.json', 'w') as f:
        f.write(regional_json)

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        use_regional=True,
        regional_json=regional_json,
        w=1526,
        h=1024,
        regional_sr_delta=0.85,
        regional_hb_replace=2.0,
    )
    prompt.text=f"<lora:{FLUX_LORA_GAME_UI_2.id}:1> monetization gaming banner. Opaque Modal with a golden 3D rim and a dark blue background with a repeating crown pattern on it, text banner at the top, button on the bottom. Bright, colorful, and vibrant mobile game modal letting the user know about a new promotion. Cartoonish and modern design on a white background."
    prompt.tunes=[FLUX_LORA_GAME_UI_2]
    run_images(prompt)
