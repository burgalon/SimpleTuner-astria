import os, sys

sys.path.append("astria")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
from astria_tests.test_infer import IMG_POSE, BASE_PROMPT, FLUX_LORA, run_images, FLUX_FACEID, pipe

# The upscaler class you placed here:
from seedvr2.upscaler import SeedVR2ImageUpscaler
from astria_utils import JsonObj, MODELS_DIR
from image_utils import load_image



def test_seedvr2_image_upscale():
    up = SeedVR2ImageUpscaler()
    up.load()
    # img = load_image('https://mp.astria.ai/ry34z8d9nnmti9eizrs066ty2yl4')
    out = up.upscale(IMG_POSE, seed=0, sample_steps=1, cfg_scale=1.0)
    out.save(MODELS_DIR + f"/test_seedvr2_image_upscale.png")

def test_seedvr2_pipe():
    pipe.reset(True)
    input_images = [load_image(t) for t in [
        'https://mp.astria.ai/ry34z8d9nnmti9eizrs066ty2yl4',
        'https://mp.astria.ai/z9bir6muno34znw1ar8f29p0ocul',
        'https://mp.astria.ai/80alvnq3rqijf7n1f5lz5xxum37v',
        'https://mp.astria.ai/p0samx6m8hnslit3l6spnzqgqlpt',
    ]]
    images = pipe.upscale_seedvr2(input_images, BASE_PROMPT)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/test_seedvr2_pipe-{i}.png")

def test_txt2img_lora_upscale_seedvr2():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        inpaint_faces=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-before-upscale')

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        w=896,
        h=1152,
        super_resolution=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    prompt.tunes=[FLUX_LORA]
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-upscaled')


    # Check post super-resolution everything is okay
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        inpaint_faces=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-after-upscale')

