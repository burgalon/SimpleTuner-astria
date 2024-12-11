import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Generic, TypeVar, Union

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageFilter
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO

from astria_utils import MODELS_DIR, JsonObj, device, HUMAN_CLASS_NAMES, CACHE_DIR
from pipeline_flux_differential_img2img import FluxDifferentialImg2ImgPipeline
from super_resolution_helper import load_sr, upscale_sr

T = TypeVar("T", int, float)

YOLO_FACE_MODEL = f"{CACHE_DIR}/face_yolov8m.pt"

@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    preview: Optional[Image.Image] = None
    image: Optional[Image.Image] = None

# https://github.com/Bing-su/adetailer/blob/03ec9d004ae2e7051506b3485a81c67d028d38e4/adetailer/common.py#L120
def create_mask_from_bbox(
        bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks

def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]

def bbox_area(bbox: list[T]) -> T:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# Filter by ratio
def is_in_ratio(bbox: list[T], low: float, high: float, orig_area: int) -> bool:
    area = bbox_area(bbox)
    return low <= area / orig_area <= high

def filter_by_ratio(
        pred: PredictOutput[T], low: float, high: float
) -> PredictOutput[T]:
    if not pred.bboxes:
        return pred

    w, h = pred.image.size
    orig_area = w * h
    items = len(pred.bboxes)
    idx = [i for i in range(items) if is_in_ratio(pred.bboxes[i], low, high, orig_area)]
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    return pred

def ultralytics_predict(
        model,
        image: Image.Image,
        confidence: float = 0.3,
        device: str = "",
        classes: str = "",
) -> PredictOutput[float]:
    # apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    bboxes = bboxes.tolist()

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)
    if os.environ.get('DEBUG', '') == 'inpaint_faces':
        preview = pred[0].plot()
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview)
        preview.save(f"{MODELS_DIR}/ultralytics_predict.jpg")
    else:
        preview = None

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview, image=image)

def filter_k_largest(pred: PredictOutput, k: int = 0) -> PredictOutput:
    if not pred.bboxes or k == 0:
        return pred
    areas = [bbox_area(bbox) for bbox in pred.bboxes]
    idx = np.argsort(areas)[-k:]
    pred.bboxes = [pred.bboxes[i] for i in idx]
    pred.points = [pred.points[i] for i in idx]
    pred.masks = [pred.masks[i] for i in idx]
    return pred


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

def offset(img: Image.Image, x: int = 0, y: int = 0) -> Image.Image:
    """
    The offset function takes an image and offsets it by a given x(→) and y(↑) value.

    Parameters
    ----------
        mask: Image.Image
            Pass the mask image to the function
        x: int
            →
        y: int
            ↑

    Returns
    -------
        PIL.Image.Image
            A new image that is offset by x and y
    """
    return ImageChops.offset(img, x, -y)

def is_all_black(img: Image.Image) -> bool:
    arr = np.array(img)
    return cv2.countNonZero(arr) == 0


def mask_preprocess(
        masks: List[Image.Image],
        kernel: int = 0,
        x_offset: int = 0,
        y_offset: int = 0,
        bboxes: List[list[float]] = None,
) -> List[Image.Image]:
    """
    The mask_preprocess function takes a list of masks and preprocesses them.
    It dilates and erodes the masks, and offsets them by x_offset and y_offset.

    Parameters
    ----------
        masks: List[Image.Image]
            A list of masks
        kernel: int
            kernel size of dilation or erosion
        x_offset: int
            →
        y_offset: int
            ↑

    Returns
    -------
        List[Image.Image]
            A list of processed masks
    """
    if not masks:
        return []

    if x_offset != 0 or y_offset != 0:
        masks = [offset(m, x_offset, y_offset) for m in masks]

    if kernel != 0:
        if kernel < 1:
            # calculate the kernel size based on the bounding box
            kernel_sizes = [int(max((bbox[2] - bbox[0]) * kernel, 1)) for bbox in bboxes]
        else:
            kernel_sizes = [kernel] * len(masks)
        masks = [dilate_erode(m, k) for m, k in zip(masks, kernel_sizes)]
        masks = [m for m in masks if not is_all_black(m)]

    return masks


def pred_preprocessing(pred: PredictOutput) -> List[Image.Image]:
    pred = filter_by_ratio(pred, low=0.003, high=0.3)
    pred = filter_k_largest(pred, k=0)
    # increase bbox y2 by 10% to include chin
    pred.bboxes = [[x1, y1, x2, y2 * 1.1] for x1, y1, x2, y2 in pred.bboxes]
    # pred = sort_bboxes(pred, SortBy.AREA)
    return mask_preprocess(
        pred.masks,
        kernel=MASK_PADDING,
        x_offset=0,
        y_offset=0,
        bboxes=pred.bboxes,
    )

INPAINT_RESOLUTION = 1024
MASK_PADDING = 0.02
class InpaintFaceMixin:
    def __init__(self):
        self.reset_yolo()

    def reset_yolo(self):
        self.yolo = None


    INPAINT_RESOLUTION = 512  # Define the inpaint resolution

    def inpaint_image(self, image: Image.Image, prompt: JsonObj, lora: JsonObj):
        pred = ultralytics_predict(
            self.yolo,
            image=image,
            confidence=0.3,
            device=device,
            classes=None,  # ad_model_classes,
        )

        masks: List[Image.Image] = pred_preprocessing(pred)
        if not masks:
            print(f"T#{prompt.tune_id} P#{prompt.id} No faces detected")
            return image

        mask_image = masks[0]
        bbox = pred.bboxes[0]  # Assuming single face; adjust if multiple faces are needed

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Calculate the width and height of the bounding box
        width = x2 - x1
        height = y2 - y1

        # Increase the width and height by 80%
        new_width = int(width * 1.8)
        new_height = int(height * 1.8)

        # Ensure the new width and height are divisible by 8
        new_width = (new_width + 7) // 32 * 32
        new_height = (new_height + 7) // 32 * 32

        # Adjust the coordinates of the bounding box
        x1 = max(0, x1 - (new_width - width) // 2)
        y1 = max(0, y1 - (new_height - height) // 2)
        x2 = x1 + new_width
        y2 = y1 + new_height

        # Crop the image and mask
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_mask = mask_image.crop((x1, y1, x2, y2))

        # Add gaussian blur to mask for smooth blending
        cropped_mask = cropped_mask.filter(ImageFilter.GaussianBlur(10))

        # Resize cropped region to INPAINT_RESOLUTION for inpainting
        k = (float(INPAINT_RESOLUTION * INPAINT_RESOLUTION) / (new_width * new_height)) ** 0.5
        new_width = int(new_width * k) // 32 * 32
        new_height = int(new_height * k) // 32 * 32

        # if we need to resize more than X2 - which is the ESRGAN outscale factor, then then face are too big
        # we should skip inpainting because it will be too blurry
        if k<0.5:
            print(f"T#{prompt.tune_id} P#{prompt.id} skipping inpainting, face too big {k}")
            return image

        print(f"T#{prompt.tune_id} P#{prompt.id} resizing {cropped_image.size} => {new_width}x {new_height} k={k}")
        cropped_image_resized = cropped_image.resize((new_width, new_height), Image.LANCZOS)
        cropped_mask_resized = cropped_mask.resize((new_width, new_height), Image.LANCZOS)

        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            cropped_mask_resized.save(f"{MODELS_DIR}/{prompt.id}-cropped-mask.jpg")
            cropped_image_resized.save(f"{MODELS_DIR}/{prompt.id}-cropped-image.jpg")

        # Invert mask for differential diffusion if necessary
        pipe = self.inpaint
        if isinstance(pipe, FluxDifferentialImg2ImgPipeline):
            print("Inverting mask for differential diffusion")
            cropped_mask_resized = ImageOps.invert(cropped_mask_resized)

        # Inpaint the resized cropped region
        inpainted_crop_resized = cropped_image_resized
        for i in range(1):
            inpainted_crop_resized = pipe(
                prompt=prompt.text,
                guidance_scale=float(prompt.cfg_scale or 2.5),
                height=cropped_image_resized.height,
                width=cropped_image_resized.width,
                num_inference_steps=28,
                max_sequence_length=prompt.max_sequence_length or 512,
                generator=torch.Generator(device="cuda").manual_seed(42),
                joint_attention_kwargs={"scale": 1.0},
                mask_image=cropped_mask_resized,
                image=inpainted_crop_resized,
                strength=prompt.face_inpaint_denoising or 0.7,
            ).images[0]
            if os.environ.get('DEBUG', '') == 'inpaint_faces':
                inpainted_crop_resized.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-{i}.jpg")

        # upscale the inpainted crop only if we resized up
        if (prompt.super_resolution or os.environ.get('SUPER_RESOLUTION')) and k>=1:
            print(f"T#{prompt.tune_id} P#{prompt.id} inpaint_faces upscaling before resize")
            if not self.sr_model:
                self.sr_model = load_sr(f"/data/cache/4x_NMKD-Siax_200k.pth")
            if os.environ.get('DEBUG') == 'inpaint_faces':
                inpainted_crop_resized.save(f"{MODELS_DIR}/{prompt.id}-before-esrgan-face-inpaint.jpg")
            inpainted_crop_resized = upscale_sr(self.sr_model, inpainted_crop_resized, 1)

        # Resize the inpainted crop back to original cropped dimensions
        inpainted_crop = inpainted_crop_resized.resize(cropped_image.size, Image.LANCZOS)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            inpainted_crop.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop.jpg")

        # upscale the inpainted crop only if we resized down
        if (prompt.super_resolution or os.environ.get('SUPER_RESOLUTION')) and k<1:
            print(f"T#{prompt.tune_id} P#{prompt.id} inpaint_faces upscaling after resize")
            if not self.sr_model:
                self.sr_model = load_sr(f"/data/cache/4x_NMKD-Siax_200k.pth")
            if os.environ.get('DEBUG') == 'inpaint_faces':
                inpainted_crop.save(f"{MODELS_DIR}/{prompt.id}-before-esrgan-face-inpaint.jpg")
            inpainted_crop = upscale_sr(self.sr_model, inpainted_crop, 1)

        # Create an alpha composite for smooth blending using the blurred mask
        inpainted_crop_with_alpha = inpainted_crop # Image.composite(inpainted_crop, cropped_image, cropped_mask)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            inpainted_crop_with_alpha.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-alpha.jpg")

        # Paste the blended result back onto the original image
        print(f"T#{prompt.tune_id} P#{prompt.id} {cropped_mask.size=} {inpainted_crop_with_alpha.size=}")
        image.paste(inpainted_crop_with_alpha, (x1, y1))
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            image.save(f"{MODELS_DIR}/{prompt.id}-inpainted-image.jpg")

        return image



    def inpaint_faces(self, images: List[Image.Image], prompt: JsonObj, tune: JsonObj):
        if not self.yolo:
            self.yolo = YOLO(YOLO_FACE_MODEL)

        lora = next(iter(t for t in prompt.tunes if t.name in HUMAN_CLASS_NAMES), None)
        if not lora:
            return images

        self.init_inpaint()
        # change weight to 1 in case its lower
        # new_weight = [w if name == str(lora.id) else w for name, w in zip(self.current_lora_weights['names'], self.current_lora_weights['scales'])]
        # self.pipe.set_adapters(self.current_lora_weights['names'], adapter_weights=self.current_lora_weights['scales'])

        for i, image in enumerate(images):
            # reuse self.inpaint_image
            if os.environ.get('DEBUG', '') == 'inpaint_faces':
                image.save(f"{MODELS_DIR}/{prompt.id}-before-inpaint-{i}.jpg")
            images[i] = self.inpaint_image(image, prompt, tune)
        return images

if __name__ == "__main__":
    import sys
    import copy
    sys.path.append("/app")
    from infer import InferPipeline, load_image
    pipe = InferPipeline()
    from astria_tests.test_infer import TUNE_FLUX, BASE_PROMPT, FLUX_LORA

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        inpaint_faces=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1.0> An elegant, noble ohwx woman in a structured, taffeta dress with a high neckline and puffed sleeves, sitting in a lavish, gilded chair."
    prompt.tunes=[FLUX_LORA]

    pipe.init_pipe(MODELS_DIR + f"/{TUNE_FLUX.id}-{TUNE_FLUX.branch}")
    image = load_image('/data/models/19306399-before-inpaint-0.jpg')
    pipe.load_references(prompt)

    images = pipe.inpaint_faces([image], prompt, TUNE_FLUX)
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/{prompt.id}-{i}.jpg")

