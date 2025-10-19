import os
import copy
from pathlib import Path

# The upscaler class you placed here:
from astria.seedvr2.upscaler import SeedVR2ImageUpscaler

from infer import *
from astria_utils import JsonObj, MODELS_DIR

from test_infer import MODELS_DIR, IMG_POSE, BASE_PROMPT, FLUX_LORA, run_images


def test_seedvr2_image_upscale():
    """
    Smoke test: loads SeedVR2 7B, upscales the pose.jpg fixture, and writes an output.

    Skips if:
      - CUDA is unavailable
      - Text embeddings (pos_emb.pt / neg_emb.pt) are not present and not provided via env vars
    """
    img_path = IMG_POSE

    # Instantiate the upscaler (use defaults for config path)
    up = SeedVR2ImageUpscaler()

    # Load models
    up.load()

    # Run upscale
    import time
    start = time.time()
    out = up.upscale(str(img_path), seed=0, sample_steps=1, cfg_scale=1.0)
    out.save(MODELS_DIR + f"/test_seedvr2_image_upscale.png")

def test_txt2img_lora_upscale_seedvr2():
    pipe = InferPipeline()
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    prompt.tunes=[FLUX_LORA]
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()
    run_images(prompt)
    assert pipe.last_pipe is None
