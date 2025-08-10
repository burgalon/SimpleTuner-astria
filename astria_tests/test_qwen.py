import copy
from test_infer import pipe, run_images, MODELS_DIR, name, FLUX_FACEID, JsonObj
from image_utils import load_image
from diffusers import QwenImagePipeline

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
    "id": 3086607,
    "name": "",
    "title": "Realism",
    "branch": "qwen-image-1",
    "token": str(3086607),
    "train_token": "",
    "model_type": "lora",
})


def test_qwen_txt2img_no_lora():
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
    )
    prompt.text=f"woman holding flowers"
    run_images(prompt)
    assert isinstance(pipe.last_pipe, QwenImagePipeline)

def test_qwen_txt2img_lora():
    prompt = JsonObj(
        **copy.copy(QWEN_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{QWEN_LORA.id}:1> woman holding flowers"
    prompt.tunes=[QWEN_LORA]
    run_images(prompt)
    assert isinstance(pipe.last_pipe, QwenImagePipeline)

