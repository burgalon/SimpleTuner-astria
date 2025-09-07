import copy
import os
import sys

from pathlib import Path

# one level app + /astria
sys.path.append("astria")

import imagehash
from PIL import Image

from infer import *
from astria_utils import JsonObj, MODELS_DIR

# Do not send to server?
if 'DEBUG' not in os.environ:
    os.environ['DEBUG'] = 'test'
    os.environ['MOCK_SERVER'] = '1'


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

FLUX_LORA_MAN = JsonObj(**{
    "id": 1979152,
    "name": "man",
    "title": "Lance",
    "branch": "flux1",
    "token": str(1979152),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": [
        "https://mp.astria.ai/wtwonr79gmb7sqep55zsae9040qn",
        "https://mp.astria.ai/4cag22nur3q2sfbecbgnrumqxra2",
        "https://mp.astria.ai/kzc0nw7jtojh81us3b9ldcxyusk0"
    ]
})

FLUX_LORA_MAN_MARCO = JsonObj(**{
    "id": 1557315,
    "name": "man",
    "title": "Marco",
    "branch": "flux1",
    "token": str(1557315),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": [
        "https://mp.astria.ai/2y388ja53947ie2ecp4k9kd95lp9",
        "https://mp.astria.ai/a544y3n2mm9xvg99ofyat947lp2w",
        "https://mp.astria.ai/11knqm2byx97er3vg950c7v6e1mt"
    ]
})

FLUX_LORA_WOMAN_2 = JsonObj(**{
    "id": 1617347,
    "name": "woman",
    "title": "Woman",
    "branch": "flux1",
    "token": str(1617347),
    "train_token": "ohwx",
    "model_type": "lora"
})

FLUX_LORA_SHOE = JsonObj(**{
    "id": 1612840,
    "name": "man",
    "title": "OnCloud brown shoe",
    "branch": "flux1",
    "token": str(1612840),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": [
        "https://mp.astria.ai/azvn52teo5km69kz28olleglqtxe",
        "https://mp.astria.ai/9p607buzsgba1ew152cz2aybcojs",
        "https://mp.astria.ai/0cbp7yk70atdst0zpxbdqmugovo1",
        "https://mp.astria.ai/zay337rw46of3zey21nlbpveo6e0",
        "https://mp.astria.ai/q5v3xxwe9phexwy4yb5ktqnlbg2r",
    ]
})

FLUX_LORA_DRESS = JsonObj(**{
    "id": 2689680,
    "name": "dress",
    "title": "Floral dress",
    "branch": "flux1",
    "token": str(2689680),
    "train_token": "floral white",
    "model_type": "lora",
    "face_swap_images": [
        "https://mp.astria.ai/plxxygl17gdl2pzbzct3o0dl2qx2",
        "https://mp.astria.ai/qb6c9jaffrjbiou9hltqqtfubohe",
        "https://mp.astria.ai/6olwfbvsn79362m7loj0kw2sqnyx",
        "https://mp.astria.ai/af23cowcprxdetl9c0vuv3d0npta",
    ]
})

FLUX_LORA_COAT = JsonObj(**{
    "id": 2018489,
    "name": "coat",
    "title": "coat",
    "branch": "flux1",
    "token": str(2018489),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": []
})

FLUX_LORA_PANTS = JsonObj(**{
    "id": 2005693,
    "name": "pants",
    "title": "pants",
    "branch": "flux1",
    "token": str(2005693),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": []
})

FLUX_FACEID = JsonObj(**{
    "id": 1533312,
    "name": "woman",
    "title": "Emma",
    "branch": "flux1",
    "token": str(1533312),
    "model_type": "faceid",
    "face_swap_images": [
        "https://mp.astria.ai/w2ra2h8m8hx6okt9jmrwm6mlmova",
        "https://mp.astria.ai/f8bg0pac6m740muuicmtlzil2nny",
        "https://mp.astria.ai/q8kr3j8qy7ma6dq8xf8aqm27v12a",
        "https://mp.astria.ai/1t3y9jvi249mn1m3s689nw9w8e9z",
        "https://mp.astria.ai/p1jhygwtgxx4pwc2cm7kmjnpw80e",
        "https://mp.astria.ai/a90eqb8jzebf0njd5gqgh23ylr83",
        "https://mp.astria.ai/py1thtz5yem8a66sfn1z06wlds1y",
        "https://mp.astria.ai/l7pp1gzy1lthev4u1fiwvdt5l9tg",
    ]
})

FLUX_CARTOON = JsonObj(**{
    "id": 1989689,
    "name": "man",
    "title": "Sloth",
    "branch": "flux1",
    "token": str(1989689),
    "train_token": "ohwx",
    "model_type": "lora",
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

FLUX_EXTERNAL_LORA = JsonObj(**{
    "id": "wow_details",
    "name": "style",
    "title": "Wow details comfyui compatible LoRA",
    "branch": "flux1",
    "token": "wow_details",
    "train_token": "ohwx",
    "model_type": "lora",
})

FLUX_EXTERNAL_LORA_2 = JsonObj(**{
    "id": "flux_realism_lora",
    "name": "style",
    "title": "flux_realism_lora",
    "branch": "flux1",
    "token": "wow_details",
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

IMG_POSE = "astria_tests/fixtures/pose.jpg"

def name():
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

test_name_invocation_count = {}

def run_images(prompt, override_name=None, base_tune=TUNE_FLUX):
    prompt.id = name() if override_name is None else override_name
    if test_name_invocation_count.get(prompt.id, None) is None:
        test_name_invocation_count[prompt.id] = 0
    else:
        test_name_invocation_count[prompt.id] += 1
    count = test_name_invocation_count[prompt.id]

    tune = JsonObj(**base_tune.__dict__, prompts=[prompt])
    images = pipe.infer(tune)

    if not prompt.video:
        for i, image in enumerate(images):
            # check if has alpha
            if image.mode == 'RGBA':
                image.save(MODELS_DIR + f"/{prompt.id}-{i}.png")
                img_fn = f"{prompt.id}-{count}-{i}.png"
            else:
                image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
                img_fn = f"{prompt.id}-{count}-{i}.jpg"
            if (Path(__file__).parent / 'results' / img_fn).exists():
                pth = MODELS_DIR + f"/{img_fn}"
                image.save(pth)
                hash_ref =  imagehash.phash(Image.open(
                    (Path(__file__).parent / 'results' / img_fn).absolute()))
                hash_out =  imagehash.phash(Image.open(pth))
                assert hash_ref == hash_out, "Image hash does not match"
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

# def test_txt2img_civitai_lora():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#     )
#     prompt.text=f"<lora:{FLUX_EXTERNAL_LORA.id}:1> {FLUX_EXTERNAL_LORA.train_token} woman holding flowers"
#     prompt.tunes=[FLUX_EXTERNAL_LORA]
#     run_images(prompt)
#     assert isinstance(pipe.last_pipe, FluxPipeline)
#
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#     )
#     prompt.text=f"<lora:{FLUX_EXTERNAL_LORA_2.id}:1> {FLUX_EXTERNAL_LORA_2.train_token} woman holding flowers"
#     prompt.tunes=[FLUX_EXTERNAL_LORA_2]
#     run_images(prompt)
#     assert isinstance(pipe.last_pipe, FluxPipeline)

def test_txt2img_after():
    run_images(BASE_PROMPT)
    assert isinstance(pipe.last_pipe, FluxPipeline)

def test_faceid():
    os.environ['FORCE_PULID'] = '1'
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<{FLUX_FACEID.model_type}:{FLUX_FACEID.id}:1> woman holding flowers"
    prompt.tunes=[FLUX_FACEID]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipelineWithPulID)

def test_bad_faceid():
    os.environ['FORCE_PULID'] = '1'
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    flux_faceid = JsonObj(**FLUX_FACEID.__dict__)
    flux_faceid.face_swap_images = [
        'https://mp.astria.ai/a93ocfwgzocdrmq1q4wizwajnhvm'
    ]
    prompt.text=f"<{FLUX_FACEID.model_type}:{FLUX_FACEID.id}:1> woman holding flowers"
    prompt.tunes=[flux_faceid]
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
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetPipeline)

# def test_controlnet_ipadapter():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#         input_image='https://mp.astria.ai/u77j61zzd4gbqdsvkhmd9p1cz0d1',
#         controlnet='ipadapter',
#     )
#     prompt.controlnet_txt2img = True
#     prompt.text=f"earrings placed on a pedestal"
#     prompt.tunes=[FLUX_LORA]
#     prompt.controlnet_conditioning_scale=0.5
#     run_images(prompt)
#     assert isinstance(pipe.last_pipe, FluxControlNetPipeline)

def test_controlnet_img2img():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        controlnet='pose',
    )
    prompt.controlnet_txt2img = False
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    prompt.denoising_strength=0.9
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetImg2ImgPipeline)


def test_film_grain():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__), film_grain=True)
    run_images(prompt)

def test_clut():
    prompt = JsonObj(**copy.copy(BASE_PROMPT.__dict__), color_grading='Film Velvia')
    run_images(prompt)

def test_leak_load_references():
    # avoid leakage from other tests
    if pipe.pipe:
        pipe.reset_controlnet()
        pipe.unload_lora_weights(pipe.pipe)
    if pipe.fill:
        pipe.unload_lora_weights(pipe.fill)

    # 1. fill
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1.4> {FLUX_LORA.train_token} woman with black blouse --mask_prompt foreground --mask_dilate 1% --fill"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, name() + '-1-lora')
    assert isinstance(pipe.last_pipe, FluxFillPipeline)
    assert len(pipe.current_lora_weights_map['pipe']['names']) == 0
    assert pipe.current_lora_weights_map['fill']['names'] ==  [FLUX_LORA.token]


    # 2. txt2img
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, name() + '-2-lora')
    assert isinstance(pipe.last_pipe, FluxPipeline)
    assert pipe.current_lora_weights_map['pipe']['names'] ==  [FLUX_LORA.token]


    # 3. txt2img with different LORA
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA_MAN.id}:1> {FLUX_LORA_MAN.train_token} man holding flowers"
    prompt.tunes=[FLUX_LORA_MAN]
    run_images(prompt, name() + '-3-different-lora')
    assert isinstance(pipe.last_pipe, FluxPipeline)
    for k, v in pipe.current_lora_weights_map.items():
        print(k.__class__.__name__, v)
    assert pipe.current_lora_weights_map['pipe']['names'] ==  [FLUX_LORA_MAN.token]


    # 4. fill without lora
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    prompt.text=f"woman with black blouse --mask_prompt foreground --mask_dilate 1% --fill"
    prompt.tunes=[]
    run_images(prompt, name() + '-4-no-lora')
    assert isinstance(pipe.last_pipe, FluxFillPipeline)
    assert pipe.current_lora_weights_map['fill']['names'] ==  []

    # 5. txt2img without lora
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.tunes=[]
    run_images(prompt, name() + '-5-no-lora')
    assert isinstance(pipe.last_pipe, FluxPipeline)
    assert pipe.current_lora_weights_map['pipe']['names'] ==  []


