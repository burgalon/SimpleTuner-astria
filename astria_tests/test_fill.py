import copy

from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, FluxFillPipeline, FluxImg2ImgPipeline


def test_fill_background_normal():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    prompt.text=f"lush forest --mask_prompt foreground --mask_invert --mask_dilate 0.5% --fill"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_background_product():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # necklace
        # input_image='https://sdbooth2-production.s3.amazonaws.com/pwe6bcgo9ykbnt6tya91z3omoog4',
        # earings
        # input_image='https://sdbooth2-production.s3.amazonaws.com/u77j61zzd4gbqdsvkhmd9p1cz0d1',
        # dogs
        input_image='https://sdbooth2-production.s3.amazonaws.com/a93ocfwgzocdrmq1q4wizwajnhvm',
    )
    # prompt.num_images = 8
    prompt.w = prompt.h = None
    prompt.text=f"studio shot, on a rock, surrounded by tropical vegetation, visible moisture in the air, water drops, over a blurry background, drama, high contrast image, teal and orange photo filter --mask_prompt background --mask_dilate 0 --mask_blur 0 --mask_inc_brightness 10 --fill"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_fill_foreground():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman with black blouse --mask_prompt foreground --mask_dilate 1% --fill"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxFillPipeline)

def test_outpaint_no_resize():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # input 576x864
        input_image='https://v3.fal.media/files/penguin/YhNkj0L4fBh-EFGeID93O_d19ac40ce3a5492492299cc5f14247e8.png',
    )
    # prompt.super_resolution = True
    prompt.denoising_strength = 0.0
    prompt.text=f"--outpaint top-center --outpaint_width 864 --outpaint_height 1296"
    # prompt.w = 576
    # prompt.h = 864
    images = run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)
    assert images[0].width == 864
    assert images[0].height == 1296

def test_outpaint_with_downscale_width():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # input 576x864
        input_image='https://v3.fal.media/files/penguin/YhNkj0L4fBh-EFGeID93O_d19ac40ce3a5492492299cc5f14247e8.png',
    )
    # prompt.super_resolution = True
    prompt.denoising_strength = 0.0
    prompt.text=f"--outpaint top-center --outpaint_width 512 --outpaint_height 1024"
    # prompt.w = 576
    # prompt.h = 864
    images = run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)
    assert images[0].width == 512
    assert images[0].height == 1024

def test_outpaint_with_downscale_height():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # input 576x864
        # input_image='https://v3.fal.media/files/penguin/YhNkj0L4fBh-EFGeID93O_d19ac40ce3a5492492299cc5f14247e8.png',
        # input_image='https://v3.fal.media/files/tiger/htI_fTnhrO0OZiHHE-Vbp_52d435fc9b60489498db3af9d88fddb9.png',
        input_image='https://sdbooth2-production.s3.amazonaws.com/iimdd2modj7r52j7x6naa12vf3g6',
    )
    # prompt.super_resolution = True
    prompt.denoising_strength = 0.0
    prompt.text=f"--outpaint top-center --outpaint_width 1024 --outpaint_height 512"
    # prompt.w = 576
    # prompt.h = 864
    images = run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)
    assert images[0].width == 1024
    assert images[0].height == 512
