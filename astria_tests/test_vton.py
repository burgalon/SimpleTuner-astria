import copy

from test_infer import pipe, BASE_PROMPT, run_images, MODELS_DIR, name, FLUX_FACEID, JsonObj, FLUX_LORA
from image_utils import load_image

FLUX_VTON_LORA = JsonObj(**{
    "id": 1781064,
    "name": "dress",
    "title": "Dress test",
    "branch": "flux1",
    "token": str(1781064),
    "train_token": None,
    "model_type": "faceid",
    "face_swap_images": [
        # front
        'https://mp.astria.ai/l4b6doh3zhwz9usplypdoywcixfn',
        # right
        # 'https://mp.astria.ai/ihjcrezh9hc0ufk632vj2mk65wpb',
    ],
})

FLUX_VTON_SHIRT = JsonObj(**{
    "id": 1781064,
    "name": "dress",
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

FLUX_VTON_SHOES = JsonObj(**{
    "id": 3161080,
    "name": "shoes",
    "title": "shoes",
    "branch": 'gemini-2',
    "token": str(3161080),
    "train_token": None,
    "model_type": "faceid",
    "face_swap_images": [
        'https://mp.astria.ai/i2cm2wnqy29gsirsbu1c4af8kgc6'
    ]
})

FLUX_VTON_BOTTLE = JsonObj(**{
    "id": 2692357,
    "name": "bottle",
    "title": "bottle",
    "branch": 'gemini-2',
    "token": str(2692357),
    "train_token": None,
    "model_type": "faceid",
    "face_swap_images": [
        'https://mp.astria.ai/6xdemd8ihohfpatuqnkv0alodusx'
    ]
})

def test_vton_1_only():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.id = name()
    prompt.tunes = [FLUX_VTON_LORA, FLUX_LORA]
    images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
    images = pipe.vton(images, prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")

def test_vton_2_shoes_only():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.id = name()
    prompt.tunes = [FLUX_VTON_BOTTLE, FLUX_LORA]
    images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
    images = pipe.vton(images, prompt)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")

# def test_vton_1_failed():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#     )
#     prompt.id = name()
#     bad_tune = JsonObj(**FLUX_VTON_LORA.__dict__, )
#     bad_tune.face_swap_images=[
#         'https://mp.astria.ai/a93ocfwgzocdrmq1q4wizwajnhvm',
#     ]
#     prompt.tunes = [bad_tune]
#     # images = [load_image('https://mp.astria.ai/t5ibi8bs69e5xm87mf31a5ikbie6')]
#     images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
#     # 384 * 493
#     # pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
#     # pipe.load_references(prompt)
#     images = pipe.vton(images, prompt)
#     # run_images(prompt)
#     for i, image in enumerate(images):
#         image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
#
# def test_vton_1_full_dress_v1():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#         inpaint_faces=True,
#     )
#     prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
#     prompt.tunes = [FLUX_VTON_SHIRT]
#     prompt.super_resolution = True
#     run_images(prompt)
#
# def test_vton_1_full_shirt():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#         inpaint_faces=True,
#     )
#     prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
#     prompt.tunes = [FLUX_VTON_SHIRT]
#     prompt.super_resolution = True
#     run_images(prompt)
#
# def test_vton_img2img_strength0():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__), input_image=FLUX_FACEID.face_swap_images[6]
#     )
#     prompt.denoising_strength = 0.1
#     prompt.text = "Woman wearing white shirt with printed flowers, fashion editorial plain white background"
#     prompt.tunes = [FLUX_VTON_SHIRT]
#     run_images(prompt)
#
# def test_vton_2():
#     prompt = JsonObj(
#         **copy.copy(BASE_PROMPT.__dict__),
#         inpaint_faces=True,
#     )
#     prompt.id = name()
#     prompt.tunes = [FLUX_VTON_SHIRT]
#     images = [load_image('astria_tests/fixtures/19477328-before-inpaint-0.jpg')]
#     images = pipe.vton(images, prompt)
#     # run_images(prompt)
#     for i, image in enumerate(images):
#         image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")
#
