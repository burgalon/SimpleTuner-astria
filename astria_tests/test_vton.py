import copy
from astria_utils import JsonObj
from test_infer import pipe, BASE_PROMPT, run_images, MODELS_DIR, name, FLUX_FACEID
from image_utils import load_image

FLUX_VTON_LORA = JsonObj(**{
    "id": 1781064,
    "name": "dress",
    "title": "Dress test",
    "branch": "flux1",
    "token": str(1781064),
    "train_token": "ohwx",
    "model_type": "lora",
    "face_swap_images": [
        # front
        'https://sdbooth2-production.s3.amazonaws.com/l4b6doh3zhwz9usplypdoywcixfn',
        # right
        # 'https://sdbooth2-production.s3.amazonaws.com/ihjcrezh9hc0ufk632vj2mk65wpb',
    ],
})

FLUX_VTON_SHIRT = JsonObj(**{
    "id": 1781064,
    "name": "shirt",
    "title": "Dress test",
    "branch": "flux1",
    "token": str(1781064),
    "train_token": "ohwx",
    "model_type": "faceid",
    "face_swap_images": [
        # shirt front
        'https://cdn.pixelbin.io/v2/janvi-fynd-972107/original/astria/rahul-mishra/printed-shirt/ghost-front.jpeg',
    ],
})


def test_vton_1_only():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.tunes = [FLUX_VTON_LORA]
    # images = [load_image('https://sdbooth2-production.s3.amazonaws.com/t5ibi8bs69e5xm87mf31a5ikbie6')]
    images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
    # 384 * 493
    # pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    # pipe.load_references(prompt)
    images = pipe.vton(images, prompt)
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")

def test_vton_1_full_dress_v1():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
    prompt.tunes = [FLUX_VTON_SHIRT]
    prompt.super_resolution = True
    prompt.hires_fix = True
    run_images(prompt)

def test_vton_1_full_dress_hires():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background --vton_hires"
    prompt.tunes = [FLUX_VTON_SHIRT]
    prompt.super_resolution = True
    prompt.hires_fix = True
    run_images(prompt)


def test_vton_1_full_shirt():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
    prompt.tunes = [FLUX_VTON_SHIRT]
    prompt.super_resolution = True
    prompt.hires_fix = True
    run_images(prompt)

def test_vton_img2img_strength0():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__), input_image=FLUX_FACEID.face_swap_images[6]
    )
    prompt.denoising_strength = 0.1
    prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
    prompt.tunes = [FLUX_VTON_SHIRT]
    run_images(prompt)

def test_vton_2():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.id = name()
    prompt.tunes = [FLUX_VTON_SHIRT]
    images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
    images = pipe.vton(images, prompt)
    # run_images(prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")

