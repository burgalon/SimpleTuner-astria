from test_infer import *

def test_inpainting_background_normal():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
    )
    prompt.text=f"lush forest --mask_prompt foreground --mask_invert --mask_dilate 0.5%"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_background_product():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # necklace
        # input_image='https://sdbooth2-production.s3.amazonaws.com/pwe6bcgo9ykbnt6tya91z3omoog4',
        # earings
        # input_image='https://sdbooth2-production.s3.amazonaws.com/u77j61zzd4gbqdsvkhmd9p1cz0d1',
        # dogs
        input_image='https://sdbooth2-production.s3.amazonaws.com/a93ocfwgzocdrmq1q4wizwajnhvm',
        denoising_strength=1,
        # cfg_scale=3,
    )
    # prompt.num_images = 8
    prompt.w = prompt.h = None
    prompt.text=f"studio shot, on a rock, surrounded by tropical vegetation, visible moisture in the air, water drops, over a blurry background, drama, high contrast image, teal and orange photo filter --mask_prompt background --mask_dilate 0 --mask_blur 0 --mask_inc_brightness 10"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_foreground():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman with black blouse --mask_prompt foreground --mask_dilate 1%"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxDifferentialImg2ImgPipeline)

def test_inpainting_controlnet_foreground():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
        controlnet='pose',
    )
    prompt.controlnet_txt2img = True
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman with black blouse --control_guidance_end 0.35 --mask_prompt foreground --mask_dilate 1%"
    prompt.tunes=[FLUX_LORA]
    prompt.controlnet_conditioning_scale=0.5
    run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxControlNetInpaintPipeline)
