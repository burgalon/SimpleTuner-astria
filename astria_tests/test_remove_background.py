import copy

from test_infer import pipe, BASE_PROMPT, run_images, IMG_POSE, FLUX_LORA, JsonObj, FluxFillPipeline, FluxImg2ImgPipeline

def test_remove_background():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # input 576x864
        # input_image='https://v3.fal.media/files/penguin/YhNkj0L4fBh-EFGeID93O_d19ac40ce3a5492492299cc5f14247e8.png',
        # input_image='https://v3.fal.media/files/tiger/htI_fTnhrO0OZiHHE-Vbp_52d435fc9b60489498db3af9d88fddb9.png',
        input_image='https://mp.astria.ai/iimdd2modj7r52j7x6naa12vf3g6',
    )
    # prompt.super_resolution = True
    prompt.denoising_strength = 0.0
    prompt.text=f"--remove_background --composite https://mp.astria.ai/22st4ylkbwfv87xucsrlumamo3yb"
    images = run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)

def test_composite_hex():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        # input 576x864
        # input_image='https://v3.fal.media/files/penguin/YhNkj0L4fBh-EFGeID93O_d19ac40ce3a5492492299cc5f14247e8.png',
        # input_image='https://v3.fal.media/files/tiger/htI_fTnhrO0OZiHHE-Vbp_52d435fc9b60489498db3af9d88fddb9.png',
        input_image='https://mp.astria.ai/iimdd2modj7r52j7x6naa12vf3g6',
    )
    # prompt.super_resolution = True
    prompt.denoising_strength = 0.0
    prompt.text=f"--composite #FF0000"
    images = run_images(prompt)
    assert isinstance(pipe.last_pipe, FluxImg2ImgPipeline)

