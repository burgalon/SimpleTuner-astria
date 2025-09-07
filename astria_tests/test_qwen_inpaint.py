import copy

from diffusers import QwenImageInpaintPipeline

from test_infer import pipe, run_images, JsonObj, IMG_POSE
from test_qwen import TUNE_QWEN, QWEN_PROMPT


def test_qwen_inpainting_background_normal():
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
        input_image=IMG_POSE,
        denoising_strength=1,
    )
    prompt.text=f"lush forest --mask_prompt foreground --mask_invert --mask_dilate 0.5%"
    run_images(prompt, base_tune=TUNE_QWEN)
    assert isinstance(pipe.last_pipe, QwenImageInpaintPipeline)

