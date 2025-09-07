import copy
from test_infer import pipe, run_images, JsonObj, IMG_POSE
from diffusers import QwenImageEditPipeline

TUNE_QWEN_EDIT = JsonObj(**{
    "id": 3123913,
    "name": None,
    "title": "QWEN edit",
    "branch": "qwen-edit-1",
    "token": "",
    "model_type": None,
})

QWEN_EDIT_PROMPT = JsonObj(**{
    "id": "test-qwen-prompt-id",
    "text": "make her hat red, keep the same face",
    "tune_id": TUNE_QWEN_EDIT.id,
    "num_images": 1,
    "tunes": [],
})



def test_qwen_edit():
    prompt = JsonObj(
        **copy.copy(QWEN_EDIT_PROMPT.__dict__),
        input_image=IMG_POSE,
    )
    run_images(prompt, base_tune=TUNE_QWEN_EDIT)
    assert isinstance(pipe.last_pipe, QwenImageEditPipeline)

