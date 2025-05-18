import os
import cv2
import numpy as np
from PIL import ImageChops, ImageFilter, Image
from image_utils import load_image
from segment_anything import SamPredictor, build_sam

from astria_utils import MODELS_DIR, device, JsonObj, CACHE_DIR
from grounded_sam.grounded_sam_helper import load_model_hf, get_masks_by_class
from hinter_helper import annotator_ckpts_path

GROUNDING_DINO_MAPPING = {
    "model": "ShilongLiu/GroundingDINO",
    "checkpoint": "groundingdino_swinb_cogcoor.pth",
    "config": "GroundingDINO_SwinB.cfg.py",
}

SAM_CHECKPOINT_MAPPING = {
    "mobile_sam": "mobile_sam.pt",
    "sam": f"{CACHE_DIR}/sam_vit_h_4b8939.pth"
}

def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)

def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)


class SamMixin:
    def __init__(self, *args, **kwargs):
        self.reset_sam()

    def reset_sam(self):
        self.groundingdino_model = None
        self.sam_predictor = None

    def init_grounded_sam(self):
        if self.groundingdino_model is not None or self.sam_predictor is not None:
            return
        ckpt_repo_id = GROUNDING_DINO_MAPPING["model"]
        ckpt_filename = GROUNDING_DINO_MAPPING["checkpoint"]
        ckpt_config_filename = GROUNDING_DINO_MAPPING["config"]
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)

        self.sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT_MAPPING['sam']).to(device))

        # MobileSAM
        # sam_checkpoint = f"/data/cache/{SAM_CHECKPOINT_MAPPING['mobile_sam']}"
        # checkpoint = torch.load(sam_checkpoint)
        # mobile_sam = setup_model()
        # mobile_sam.load_state_dict(checkpoint, strict=True)
        # mobile_sam.to(device=device)
        # self.sam_predictor = SamPredictor(mobile_sam)


    def parse_sam_mask(self, prompt: JsonObj):
        if not prompt.input_image or not prompt.mask_prompt:
            return

        if self.groundingdino_model is None or self.sam_predictor is None:
            self.init_grounded_sam()

        # Also set input_image to avoid needing to load the image from network again in controlnet scenario
        prompt.input_image = input_image = load_image(prompt.input_image)

        mask = get_masks_by_class(
            input_image,
            prompt.mask_prompt,
            prompt.mask_negative,
            prompt.mask_index,
            self.groundingdino_model,
            self.sam_predictor
        )
        if mask:
            if prompt.mask_dilate:
                # check if prompt.mask_dilate is a percentage string, then calculate percentage of image size
                if prompt.mask_dilate.endswith("%"):
                    prompt.mask_dilate = int(float(prompt.mask_dilate[:-1]) / 100 * min(input_image.width, input_image.height))
                else:
                    prompt.mask_dilate = int(prompt.mask_dilate)

                mask = dilate_erode(mask, prompt.mask_dilate)

            if prompt.mask_invert:
                mask = ImageChops.invert(mask)
            if prompt.mask_blur:
                mask = mask.filter(ImageFilter.GaussianBlur(prompt.mask_blur))

            prompt.mask_image = mask
            if os.environ.get('DEBUG'):
                print(f"saving mask to {MODELS_DIR}/{prompt.id}-mask.png")
                prompt.mask_image.save(f"{MODELS_DIR}/{prompt.id}-mask.png")

        # Cleanup memory because SAM is a memory hog
        self.reset_sam()

if __name__ == "__main__":
    # Test the helper
    sam_mixin = SamMixin()
    sam_mixin.parse_sam_mask(JsonObj(
        id="test",
        input_image='https://sdbooth2-production.s3.amazonaws.com/xhthbyjell8cw2py1htv3xfc0rl5',
        mask_prompt='person',
        mask_dilate='5%',
        mask_blur=2,
    ))

    print("SAM initialized successfully.")
