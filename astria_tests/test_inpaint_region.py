import copy
from PIL import Image, ImageDraw
import numpy as np

from test_infer import (
    pipe,
    TUNE_FLUX,
    FLUX_LORA_RING,
    BASE_PROMPT,
    run_images,  # not used here, but mirrors your file import style
    MODELS_DIR,
    name,
    JsonObj,
    FluxPipeline,
    device,
)
from image_utils import load_image


IMG_RING = "astria_tests/fixtures/ring.jpg"


def _compare_images(a: Image.Image, b: Image.Image, mae_thresh=0.5):
    """
    Asserts that two images are the same size and their
    mean absolute error (0..255 scale) is <= mae_thresh.
    """
    assert a.size == b.size, f"image sizes differ: {a.size} vs {b.size}"
    A = np.asarray(a.convert("RGB"), dtype=np.uint8).astype(np.float32)
    B = np.asarray(b.convert("RGB"), dtype=np.uint8).astype(np.float32)
    mae = np.mean(np.abs(A - B))
    assert mae <= mae_thresh, f"images differ (MAE={mae:.3f} > {mae_thresh})"


def get_kwargs():
    # Encode text embeds so Path A & B use the same text.
    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = pipe.pipe.encode_prompt(
        "ohwx ring",
        "ohwx ring",
        max_sequence_length=512,
        device=device,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def test_inpaint_region_with_mask_prompt():
    """
    Uses Grounded-SAM via prompt.mask_prompt to auto-detect a region to inpaint.
    Also compares internal mixin path vs run_images() path.
    """
    # Common init
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")

    # Load source once
    src = load_image(IMG_RING)

    # ----- Path A: call the internal mixin directly -----
    prompt_A = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt_A.id = name()
    prompt_A.text = f"<lora:{FLUX_LORA_RING.id}:1> {FLUX_LORA_RING.train_token} ring"
    prompt_A.tunes = [FLUX_LORA_RING]
    prompt_A.mask_prompt = "ring (jewelry)"
    prompt_A.seed = 42
    prompt_A.num_images = 1

    # Ensure LoRA is loaded for A only (no shared mutation)
    pipe.load_references(prompt_A, pipe.pipe)

    images_A = pipe.inpaint_regions([src.copy()], prompt_A, get_kwargs())

    # Save artifacts (keeps your current behavior)
    for i, image in enumerate(images_A):
        image.save(MODELS_DIR + f"/{prompt_A.id}-{i}.png")

    # ----- Path B: call run_images (goes through full infer()) -----
    prompt_B = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt_B.id = name()
    prompt_B.text = f"<lora:{FLUX_LORA_RING.id}:1> {FLUX_LORA_RING.train_token} ring"
    prompt_B.tunes = [FLUX_LORA_RING]
    prompt_B.mask_prompt = "ring (jewelry)"
    prompt_B.seed = 42
    prompt_B.num_images = 1

    # For run_images, ensure the pipeline knows the input image and regional crop flow
    prompt_B.input_image = IMG_RING
    prompt_B.mask_crop = True

    # Deterministic alternate name so A-path files don't collide
    images_B = run_images(prompt_B, override_name=f"{prompt_B.id}-run")

    # ----- Compare outputs -----
    for a, b in zip(images_A, images_B):
        _compare_images(a, b)


def test_inpaint_region_with_mask_image():
    """
    Lets the test specify the exact region via a binary mask image.
    Compares internal mixin path vs run_images() path.
    """
    # Common init
    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")

    # Load source to size our mask
    src = load_image(IMG_RING)
    W, H = src.size

    # Build a simple centered rectangle mask
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    bw, bh = int(W * 0.1), int(H * 0.1)
    x1 = (W - bw) // 2 + 96
    y1 = (H - bh) // 2 + 96
    x2 = x1 + bw + 96
    y2 = y1 + bh + 96
    draw.rectangle([x1, y1, x2, y2], fill=255)

    # ----- Path A: internal mixin -----
    prompt_A = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt_A.id = name()
    prompt_A.text = f"<lora:{FLUX_LORA_RING.id}:1> {FLUX_LORA_RING.train_token} ring"
    prompt_A.tunes = [FLUX_LORA_RING]
    prompt_A.mask_image = mask
    prompt_A.seed = 42
    prompt_A.num_images = 1

    # Ensure LoRA is loaded for A only (no shared mutation)
    pipe.load_references(prompt_A, pipe.pipe)

    images_A = pipe.inpaint_regions([src.copy()], prompt_A, get_kwargs())

    for i, image in enumerate(images_A):
        image.save(MODELS_DIR + f"/{prompt_A.id}-{i}.png")

    # ----- Path B: run_images -----
    prompt_B = JsonObj(**copy.copy(BASE_PROMPT.__dict__))
    prompt_B.id = name()
    prompt_B.text = f"<lora:{FLUX_LORA_RING.id}:1> {FLUX_LORA_RING.train_token} ring"
    prompt_B.tunes = [FLUX_LORA_RING]
    prompt_B.mask_image = mask
    prompt_B.seed = 42
    prompt_B.num_images = 1

    prompt_B.input_image = IMG_RING
    prompt_B.mask_crop = True

    images_B = run_images(prompt_B, override_name=f"{prompt_B.id}-run")

    # ----- Compare outputs -----
    for a, b in zip(images_A, images_B):
        _compare_images(a, b)
