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

FLUX_LORA_MAN = JsonObj(**{
    "id": 1979152,
    "name": "man",
    "title": "Lance",
    "branch": "flux1",
    "token": str(1979152),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": [
        "https://sdbooth2-production.s3.amazonaws.com/wtwonr79gmb7sqep55zsae9040qn",
        "https://sdbooth2-production.s3.amazonaws.com/4cag22nur3q2sfbecbgnrumqxra2",
        "https://sdbooth2-production.s3.amazonaws.com/kzc0nw7jtojh81us3b9ldcxyusk0"
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
        "https://sdbooth2-production.s3.amazonaws.com/2y388ja53947ie2ecp4k9kd95lp9",
        "https://sdbooth2-production.s3.amazonaws.com/a544y3n2mm9xvg99ofyat947lp2w",
        "https://sdbooth2-production.s3.amazonaws.com/11knqm2byx97er3vg950c7v6e1mt"
    ]
})

FLUX_FACEID = JsonObj(**{
    "id": 1533312,
    "name": "woman",
    "title": "Emma",
    "branch": "flux1",
    "token": str(1533312),
    "model_type": "faceid",
    "face_swap_images": [
        "https://sdbooth2-production.s3.amazonaws.com/w2ra2h8m8hx6okt9jmrwm6mlmova",
        "https://sdbooth2-production.s3.amazonaws.com/f8bg0pac6m740muuicmtlzil2nny",
        "https://sdbooth2-production.s3.amazonaws.com/q8kr3j8qy7ma6dq8xf8aqm27v12a",
        "https://sdbooth2-production.s3.amazonaws.com/1t3y9jvi249mn1m3s689nw9w8e9z",
        "https://sdbooth2-production.s3.amazonaws.com/p1jhygwtgxx4pwc2cm7kmjnpw80e",
        "https://sdbooth2-production.s3.amazonaws.com/a90eqb8jzebf0njd5gqgh23ylr83",
        "https://sdbooth2-production.s3.amazonaws.com/py1thtz5yem8a66sfn1z06wlds1y",
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

def run_images(prompt, override_name=None):
    prompt.id = name() if override_name is None else override_name
    tune = JsonObj(**TUNE_FLUX.__dict__, prompts=[prompt])
    images = pipe.infer(tune)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
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
        'https://sdbooth2-production.s3.amazonaws.com/a93ocfwgzocdrmq1q4wizwajnhvm'
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
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --control_guidance_end 0.35"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetPipeline)

# def test_controlnet_ipadapter():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#         input_image='https://sdbooth2-production.s3.amazonaws.com/u77j61zzd4gbqdsvkhmd9p1cz0d1',
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
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --control_guidance_end 0.35"
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

    # 1. fill
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1.4> {FLUX_LORA.train_token} woman with black blouse --mask_prompt foreground --mask_dilate 1% --fill"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, name() + '-1-lora')
    assert isinstance(pipe.last_pipe, FluxFillPipeline)
    assert len(pipe.current_lora_weights_map.keys()) == 1
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
    prompt.text=f"man holding flowers"
    prompt.tunes=[]
    run_images(prompt, name() + '-5-no-lora')
    assert isinstance(pipe.last_pipe, FluxPipeline)
    assert pipe.current_lora_weights_map['pipe']['names'] ==  []


