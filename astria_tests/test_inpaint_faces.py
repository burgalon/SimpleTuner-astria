import copy
from astria_utils import JsonObj
from test_infer import pipe, TUNE_FLUX, FLUX_LORA, BASE_PROMPT, run_images, IMG_POSE, MODELS_DIR, name
from image_utils import load_image

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
    pipe.load_references(prompt)
    images = pipe.inpaint_faces(images, prompt, TUNE_FLUX)
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
    pipe.load_references(prompt)
    images = pipe.inpaint_faces(images, prompt, TUNE_FLUX)
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
    pipe.load_references(prompt)
    images = pipe.inpaint_faces(images, prompt, TUNE_FLUX)
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
