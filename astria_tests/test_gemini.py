import copy
import re

from test_infer import pipe, TUNE_FLUX, FLUX_LORA, BASE_PROMPT, run_images, IMG_POSE, MODELS_DIR, name, JsonObj, FluxPipeline, device, GEMINI_2_BRANCH
from partner_pipelines import PARTNER_GEMINI_TUNE_ID
from image_utils import load_image

TUNE_GEMINI = JsonObj(**{
    "id": PARTNER_GEMINI_TUNE_ID,
    "name": None,
    "title": "Gemini",
    "branch": GEMINI_2_BRANCH,
    "token": "",
    "model_type": None,
})

def test_gemini():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.tune_id = PARTNER_GEMINI_TUNE_ID
    prompt.id = name()
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, base_tune=TUNE_GEMINI)
