import copy
import re

from test_infer import pipe, TUNE_FLUX, FLUX_LORA, BASE_PROMPT, run_images, IMG_POSE, MODELS_DIR, name, JsonObj, FluxPipeline, device
from image_utils import load_image

def get_kwargs(text):
    text = re.sub(r'(<[^>]*>)', '', text)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = pipe.pipe.encode_prompt(
        text,
        text,
        max_sequence_length=512,
        device=device,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }

def test_inpaint_faces_1():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    images = [load_image(IMG_POSE)]
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    pipe.load_references(prompt, pipe.pipe)
    images = pipe.inpaint_faces(images, prompt, get_kwargs(prompt.text))
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.png")

def test_inpaint_faces_gemini():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.text=f"beautiful woman"
    # prompt.text=f"<lora:{FLUX_LORA_IRIT.id}:1> {FLUX_LORA.train_token} woman face <lora:{FLUX_LORA_SAMSUNG.id}:1> {FLUX_LORA_SAMSUNG.train_token}"
    # prompt.text=f"beautiful realistic woman face <lora:{FLUX_LORA_SAMSUNG.id}:1> {FLUX_LORA_SAMSUNG.train_token}"
    prompt.tunes=[FLUX_LORA] # , FLUX_LORA_SAMSUNG]
    images = [
        load_image('https://mp.astria.ai/jto94lmqaf3963958a09h99hwg5v'),
    ]
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    pipe.load_references(prompt, pipe.pipe)
    # pipe.load_references(prompt, pipe.pipe)
    images = pipe.inpaint_faces(images, prompt, get_kwargs(prompt.text))
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.png")

def test_inpaint_faces_2_esrgan_before_resize_big_face():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    images = [load_image('astria_tests/fixtures/19477436-before-inpaint-0.jpg')]
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    pipe.load_references(prompt, pipe.pipe)
    images = pipe.inpaint_faces(images, prompt, get_kwargs(prompt.text))
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.png")

def test_inpaint_faces_3_esrgan_after_resize_small_face():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    pipe.load_references(prompt, pipe.pipe)
    images = pipe.inpaint_faces(images, prompt, get_kwargs(prompt.text))
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.png")

def test_inpaint_faces_full():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
        w=768,
        height=1280,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman, blue hat, wearing jeans and red Tom's shoes"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxPipeline)
