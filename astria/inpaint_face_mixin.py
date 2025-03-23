import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Generic, TypeVar, Union, Any, Dict

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageFilter
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO

from astria_utils import MODELS_DIR, JsonObj, device, HUMAN_CLASS_NAMES, CACHE_DIR
from birefnet.BiRefNet_node import BiRefNet_node
from face_masking import FaceMaskGenerator, visualize_parsing
from pipeline_flux_differential_img2img import FluxDifferentialImg2ImgPipeline
from super_resolution_helper import load_sr, upscale_sr

T = TypeVar("T", int, float)

MAX_BBOX_RATIO = 0.10
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
    pred = filter_by_ratio(pred, low=0.003, high=MAX_BBOX_RATIO)
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


def feather_mask_sdf(mask_pil: Image.Image, feather_radius: int = 200, upscale_factor: int = 4) -> Image.Image:
    """
    Creates a smoothly feathered version of a binary mask by computing a signed distance function (SDF)
    from the mask boundary. This method first upscales the mask for higher resolution (to reduce jaggedness),
    computes the SDF, maps the distances to opacity values with a linear falloff over `feather_radius`,
    and then downsamples back to the original resolution.

    Args:
        mask_pil (Image.Image): A PIL image in mode 'L' (grayscale) with binary values (0 or 255).
        feather_radius (int): The radius (in pixels) over which to linearly interpolate the opacity.
        upscale_factor (int): Factor by which to upsample the mask for smoother results.

    Returns:
        Image.Image: A feathered mask as a PIL image (mode 'L').
    """
    # Upscale the mask to reduce boundary aliasing
    width, height = mask_pil.size
    up_width, up_height = width * upscale_factor, height * upscale_factor
    mask_large = mask_pil.resize((up_width, up_height), Image.NEAREST)
    mask_np = np.array(mask_large).astype(np.uint8)

    # Ensure we have a binary mask (0 and 255)
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Compute the distance transform for the background (inverted mask)
    dist_out = cv2.distanceTransform(cv2.bitwise_not(binary_mask), cv2.DIST_L2, 5)
    # Compute the distance transform for the foreground (mask)
    dist_in = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Compute the signed distance:
    # Negative values inside the mask, positive outside.
    sdf = dist_out - dist_in

    # Map the SDF to alpha values:
    # Pixels with sdf <= -feather_radius are fully inside -> alpha = 255.
    # Pixels with sdf >= feather_radius are fully outside -> alpha = 0.
    # Pixels in between are linearly interpolated.
    alpha = np.zeros_like(sdf, dtype=np.float32)
    alpha[sdf <= -feather_radius] = 255
    alpha[sdf >= feather_radius] = 0
    # Linear interpolation for distances in between.
    zone = (sdf > -feather_radius) & (sdf < feather_radius)
    alpha[zone] = (1 - ((sdf[zone] + feather_radius) / (2 * feather_radius))) * 255

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # Downsample the mask back to the original resolution
    feathered_mask = Image.fromarray(alpha, mode='L').resize((width, height), Image.LANCZOS)
    return feathered_mask

def feather_mask_outwards(mask_pil, feather_radius=400, upscale_factor=4):
    """
    Keeps the original mask fully opaque (255), and only feathers outward
    into areas that were originally 0 (background).
    """
    # Upscale the mask
    width, height = mask_pil.size
    up_width, up_height = width * upscale_factor, height * upscale_factor
    mask_large = mask_pil.resize((up_width, up_height), Image.NEAREST)
    mask_np = np.array(mask_large).astype(np.uint8)

    # Binary threshold (0 or 255)
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Distance transform of the *inverted* mask
    # => distance to the nearest 'inside' pixel
    dist = cv2.distanceTransform(cv2.bitwise_not(binary_mask), cv2.DIST_L2, 5)

    # Create alpha array fully at 255
    alpha = np.ones_like(dist, dtype=np.float32) * 255

    # For pixels outside the mask (where original was 0),
    # let alpha fall off from 255 => 0 as dist goes from 0 => feather_radius
    #  alpha_out = 255 - (dist / feather_radius) * 255
    alpha_out = 255 - (dist / feather_radius) * 255
    alpha_out = np.clip(alpha_out, 0, 255)

    # Fill those outside pixels
    outside_mask = (binary_mask == 0)
    alpha[outside_mask] = alpha_out[outside_mask]

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    # Downsample back
    feathered_mask = Image.fromarray(alpha, mode='L')
    feathered_mask = feathered_mask.resize((width, height), Image.LANCZOS)
    return feathered_mask

def restore_colors(original: Image.Image, inpainted: Image.Image):
    from skimage.exposure import match_histograms
    original_rgb = np.array(original)
    inpainted_rgb = np.array(inpainted)
    matched = match_histograms(inpainted_rgb, original_rgb, channel_axis=-1)
    return Image.fromarray(matched)

INPAINT_RESOLUTION = 1024
MASK_PADDING = 0.02
class InpaintFaceMixin:
    def __init__(self):
        self.reset_yolo()
        self.face_masker = FaceMaskGenerator(device=device, model_root=CACHE_DIR)
        self.birefnet = BiRefNet_node()

    def reset_yolo(self):
        self.yolo = None

    def remove_background_for_inpaint_crop(self, images):
        for i_image, image in enumerate(images):
            # Get the alpha matte as a tensor and convert it to a NumPy array.
            alpha_tensor = self.birefnet.matting(image, device)
            alpha = alpha_tensor.squeeze().cpu().numpy()

            # Create a PIL image for the alpha channel (values scaled 0-255)
            alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))

            # Make sure the original image is in RGBA mode to hold the alpha channel.
            image = image.convert("RGBA")
            image.putalpha(alpha_img)

            # Create a white background image (RGBA)
            black_bg = Image.new("RGBA", image.size, (0, 0, 0, 255))

            # Composite the image with the white background using the alpha channel
            composite = Image.alpha_composite(black_bg, image)

            # Convert back to RGB (3 channels) to remove the alpha channel.
            composite = composite.convert("RGB")

            images[i_image] = composite

        return images

    INPAINT_RESOLUTION = 512  # Define the inpaint resolution

    def inpaint_image(self, image: Image.Image, prompt: JsonObj, kwargs: Dict) -> Image.Image:
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

        # if we need to resize more than X2 - which is the ESRGAN outscale factor, then face are too big
        # we should skip inpainting because it will be too blurry
        if k<0.5:
            print(f"T#{prompt.tune_id} P#{prompt.id} skipping inpainting, face too big k={k:.2f}")
            return image

        print(f"T#{prompt.tune_id} P#{prompt.id} resizing {cropped_image.size} => {new_width}x {new_height} k={k:.2f}")
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

        bbox_ratio = bbox_area(bbox) / (image.width * image.height)
        if prompt.face_inpaint_denoising:
            strength = prompt.face_inpaint_denoising
        else:
            strength = 0.4 + 0.2 * (1-min(1, (bbox_ratio-0.03) / MAX_BBOX_RATIO))
        print(f"T#{prompt.tune_id} P#{prompt.id} {bbox_ratio=:.4f} {strength=:.2f}")

        # Inpaint the resized cropped region
        inpainted_crop_resized = cropped_image_resized
        for i in range(1):
            inpainted_crop_resized = pipe(
                prompt_embeds=kwargs.get('prompt_embeds'),
                pooled_prompt_embeds=kwargs.get('pooled_prompt_embeds'),
                guidance_scale=float(prompt.cfg_scale or 2.5),
                height=cropped_image_resized.height,
                width=cropped_image_resized.width,
                num_inference_steps=28,
                max_sequence_length=prompt.max_sequence_length or 512,
                generator=torch.Generator(device=device).manual_seed(42),
                joint_attention_kwargs={"scale": 1.0},
                mask_image=cropped_mask_resized,
                image=inpainted_crop_resized,
                strength=strength,
            ).images[0]
            if os.environ.get('DEBUG', '') == 'inpaint_faces':
                inpainted_crop_resized.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-{i}.jpg")

        inpainted_crop_with_alpha = restore_colors(cropped_image_resized, inpainted_crop_resized)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            inpainted_crop_with_alpha.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-colors.jpg")

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

        bg_removed = self.remove_background_for_inpaint_crop([inpainted_crop])[0]
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            bg_removed.save(f"{MODELS_DIR}/{prompt.id}-bg-removed.jpg")

        try:
            mask_np, face_map, face_orig_crop_image = self.face_masker.get_face_mask(
                bg_removed,
                include_neck=not prompt.face_inpaint_exclude_neck,
            )
            if os.environ.get('DEBUG', '') == 'inpaint_faces':
                debug_img = visualize_parsing(face_map, face_orig_crop_image)
                debug_img.save(f"{MODELS_DIR}/{prompt.id}-face-map.jpg")
        except ValueError as e:
            # this is a unique case where Yolo detected faces but the face masker failed
            # "No faces found in the image."
            print(f"T#{prompt.tune_id} P#{prompt.id} get_face_mask failed: {e}")
            return image

        # Convert the NumPy mask (with values 0 or 255) into a PIL image.
        # Resize it to match the dimensions of the final inpainted crop.
        mask_pil = Image.fromarray(mask_np, mode='L')
        if mask_pil.size != inpainted_crop.size:
            mask_pil = mask_pil.resize(inpainted_crop.size, Image.LANCZOS)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            mask_pil.save(f"{MODELS_DIR}/{prompt.id}-crop-image-original.jpg")

        # Optionally, blur the mask to soften the edges.
        # mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=10))
        mask_blurred = feather_mask_outwards(mask_pil)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            mask_blurred.save(f"{MODELS_DIR}/{prompt.id}-crop-image-blurred.jpg")

        # upscale the inpainted crop only if we resized down
        if (prompt.super_resolution or os.environ.get('SUPER_RESOLUTION')) and k<1:
            print(f"T#{prompt.tune_id} P#{prompt.id} inpaint_faces upscaling after resize")
            if not self.sr_model:
                self.sr_model = load_sr(f"/data/cache/4x_NMKD-Siax_200k.pth")
            if os.environ.get('DEBUG') == 'inpaint_faces':
                inpainted_crop.save(f"{MODELS_DIR}/{prompt.id}-before-esrgan-face-inpaint.jpg")
            inpainted_crop = upscale_sr(self.sr_model, inpainted_crop, 1)

        # Create an alpha composite for smooth blending using the blurred mask
        # inpainted_crop_with_alpha = inpainted_crop # Image.composite(inpainted_crop, cropped_image, cropped_mask)
        # if os.environ.get('DEBUG', '') == 'inpaint_faces':
        #     inpainted_crop_with_alpha.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-alpha.jpg")

        # 5. Composite the inpainted crop with the original cropped image using the face mask.
        #    In this example, pixels where mask_blurred is closer to white (255) will come from inpainted_crop,
        #    and where it is black (0) will come from the original cropped_image.
        inpainted_crop_with_alpha = Image.composite(inpainted_crop, cropped_image, mask_blurred)
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            inpainted_crop_with_alpha.save(f"{MODELS_DIR}/{prompt.id}-inpainted-crop-alpha.jpg")

        # 6. Paste the blended result back onto the original image.
        # image.paste(blended_crop, (x1, y1))

        # Paste the blended result back onto the original image
        print(f"T#{prompt.tune_id} P#{prompt.id} {cropped_mask.size=} {inpainted_crop_with_alpha.size=}")
        image.paste(inpainted_crop_with_alpha, (x1, y1))
        if os.environ.get('DEBUG', '') == 'inpaint_faces':
            image.save(f"{MODELS_DIR}/{prompt.id}-inpainted-image.jpg")

        return image

    def inpaint_faces(self, images: List[Image.Image], prompt: JsonObj, kwargs: Dict):
        if not self.yolo:
            self.yolo = YOLO(YOLO_FACE_MODEL)

        lora = next(iter(t for t in prompt.tunes if t.name in HUMAN_CLASS_NAMES), None)
        if not lora:
            return images

        self.init_inpaint(prompt)

        # change weight to 1 in case its lower
        # new_weight = [w if name == str(lora.id) else w for name, w in zip(self.current_lora_weights['names'], self.current_lora_weights['scales'])]
        # self.pipe.set_adapters(self.current_lora_weights['names'], adapter_weights=self.current_lora_weights['scales'])

        for i, image in enumerate(images):
            # reuse self.inpaint_image
            if os.environ.get('DEBUG', '') == 'inpaint_faces':
                image.save(f"{MODELS_DIR}/{prompt.id}-before-inpaint-{i}.jpg")
            images[i] = self.inpaint_image(image, prompt, kwargs)
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

