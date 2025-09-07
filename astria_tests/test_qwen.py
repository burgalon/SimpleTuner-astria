import copy

from diffusers import QwenImagePipeline

from test_infer import pipe, run_images, JsonObj, IMG_POSE

TUNE_QWEN = JsonObj(**{
    "id": 3086296,
    "name": None,
    "title": "QWEN image",
    "branch": "qwen-image-1",
    "token": "",
    "model_type": None,
})

QWEN_PROMPT = JsonObj(**{
    "id": "test-qwen-prompt-id",
    "text": "woman holding flowers",
    "tune_id": TUNE_QWEN.id,
    "num_images": 1,
    "tunes": [],
})

QWEN_LORA = JsonObj(**{
    "id": 3123654,
    "name": "woman",
    "title": "woman",
    "branch": "qwen-image-1",
    "token": str(3123654),
    "train_token": "Joan",
    "model_type": "lora",
})


def test_qwen_txt2img_no_lora():
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
    )
    run_images(prompt, base_tune=TUNE_QWEN)
    assert isinstance(pipe.last_pipe, QwenImagePipeline)

def test_qwen_img2img_no_lora():
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    run_images(prompt, base_tune=TUNE_QWEN)
    assert isinstance(pipe.last_pipe, QwenImagePipeline)

def test_qwen_txt2img_lora():
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
        inpaint_faces=True,
        super_resolution=True,
    )
    prompt.inpaint_faces = True
    prompt.text=f"<lora:{QWEN_LORA.id}:1> Joan in a black short-sleeve t-shirt paired with high-waisted white wide-leg trousers gazes intently at the camera, her expression thoughtful and warm smile, her body language conveying a sense of focus and determination. set against a softly blurred background of a modern planty office environment"
    prompt.tunes=[QWEN_LORA]
    run_images(prompt, base_tune=TUNE_QWEN)
    assert isinstance(pipe.last_pipe, QwenImagePipeline)

