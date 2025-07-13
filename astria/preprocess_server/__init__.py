# File: processor_api.py

import asyncio
import base64
import io
from typing import Dict, Optional, Tuple

import aiohttp
import cv2
import numpy as np
import torch
import uvicorn
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

# NOTE: Make sure these local modules are in your Python path.
from astria.face_masking import FaceMaskGenerator
from birefnet.BiRefNet_node import BiRefNet_node
from inpaint_face_mixin import (
    ultralytics_predict,
    filter_by_ratio,
    filter_k_largest,
    YOLO_FACE_MODEL,
)

BASE_PREPROCESS_PORT = 10398

# ----------------------------------------------------------------------
# ─── MODELS & APP INITIALIZATION ───────────────────────────────────────
# ----------------------------------------------------------------------

class PreprocessingConfig(BaseModel):
    image_url: str
    resolution: int = 512
    segmentation: bool = True
    face_crop: bool = True
    only_face: bool = False
    black_mask: bool = False
    is_human_subject: bool = True
    use_bisenet: bool = False
    crop_expansion_factor: float = 0.0

class ProcessedImagePayload(BaseModel):
    blur_factor: Optional[float]
    skipped: bool
    images: Dict[str, str]

app = FastAPI(title="Image Processing Service")

# Global variables to hold the loaded models
birefnet: Optional[BiRefNet_node] = None
yolo: Optional[YOLO] = None

@app.on_event("startup")
async def startup_event():
    """Load heavy AI models once on startup."""
    global birefnet, yolo, face_masker
    print("🚀 Loading AI models...")
    # Ensure models are loaded to the correct device
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    birefnet = BiRefNet_node()
    yolo = YOLO(YOLO_FACE_MODEL)
    face_masker = FaceMaskGenerator(device='cuda') 
    print("✅ Models loaded successfully.")

# ----------------------------------------------------------------------
# ─── CONSOLIDATED HELPER FUNCTIONS ────────────────────────────────────
# ----------------------------------------------------------------------

Bbox = Tuple[float, float, float, float]

def infer_background(image: Image.Image) -> np.ndarray:
    """Performs background matting using BiRefNet."""
    alpha_tensor: torch.Tensor = birefnet.matting(image, "cuda")
    return alpha_tensor.squeeze().cpu().numpy()

def get_birefnet_mask(image: Image.Image) -> Image.Image:
    """Generates a binary PIL mask using BiRefNet."""
    # This function's content remains the same
    alpha = infer_background(image)
    mask_array = (alpha >= 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask_array, 'L')

def get_face_bbox(image: Image.Image, resolution: int) -> Tuple[Optional[Bbox], Optional[Bbox]]:
    """Detects the largest face and returns expanded bounding boxes."""
    pred = ultralytics_predict(yolo, image=image, confidence=0.3, device="cuda", classes=None)

    # Prepare the `pred` object to be compatible with the rigid `filter_k_largest` function.
    # If .points or .masks don't exist, we add them as lists of Nones of the correct
    # length so the subsequent list comprehensions do not crash.
    if not hasattr(pred, 'points') or pred.points is None:
        setattr(pred, 'points', [None] * len(pred.bboxes))
    if not hasattr(pred, 'masks') or pred.masks is None:
        pred.masks = [None] * len(pred.bboxes)

    pred = filter_by_ratio(pred, low=0.003, high=1)
    
    # The version of filter_k_largest you provided is now safe to call.
    # It expects k=0 to return all, but your version uses k=1 for the largest.
    # We will assume k=1 is the desired behavior for face detection.
    pred = filter_k_largest(pred, k=1) 

    if not pred.bboxes:
        return None, None

    # (The rest of the function remains the same)
    x1, y1, x2, y2 = pred.bboxes[0]
    orig_bbox = (x1 - (x2 - x1) * 0.2, y1 - (y2 - y1) * 0.2, x2 + (x2 - x1) * 0.2, y2 + (y2 - y1) * 0.1)
    expanded_bbox = (x1 - (x2 - x1) * 0.4, y1 - (y2 - y1) * 0.5, x2 + (x2 - x1) * 0.4, y2 + (y2 - y1) * 0.5)

    if expanded_bbox[2] - expanded_bbox[0] < resolution:
        diff = resolution - (expanded_bbox[2] - expanded_bbox[0])
        expanded_bbox = (expanded_bbox[0] - diff / 2, expanded_bbox[1], expanded_bbox[2] + diff / 2, expanded_bbox[3])
    if expanded_bbox[3] - expanded_bbox[1] < resolution:
        diff = resolution - (expanded_bbox[3] - expanded_bbox[1])
        expanded_bbox = (expanded_bbox[0], expanded_bbox[1] - diff / 2, expanded_bbox[2], expanded_bbox[3] + diff / 2)

    return expanded_bbox, orig_bbox


def crop_to_square(image: Image.Image, mask: Image.Image, resolution: int, expansion_factor: float) -> Tuple[Image.Image, Image.Image]:
    """Crops the image and mask to the bounding box of the mask, making it square."""
    if not mask.getbbox():
        return image, mask

    # --- MODIFICATION START ---
    # Get the tight bounding box and expand it by the factor
    bbox = list(mask.getbbox())
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Calculate expansion in pixels
    expand_x = int(width * expansion_factor)
    expand_y = int(height * expansion_factor)
    
    # Apply expansion, ensuring coordinates stay within image bounds
    bbox[0] = max(0, bbox[0] - expand_x)
    bbox[1] = max(0, bbox[1] - expand_y)
    bbox[2] = min(image.width, bbox[2] + expand_x)
    bbox[3] = min(image.height, bbox[3] + expand_y)
    # --- MODIFICATION END ---
    
    # The rest of the logic for making the box square remains the same
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if width > height:
        diff = width - height
        bbox[1] -= diff // 2
        bbox[3] += diff // 2
    elif height > width:
        diff = height - width
        bbox[0] -= diff // 2
        bbox[2] += diff // 2

    # Ensure minimum resolution, expanding if necessary
    if (bbox[2] - bbox[0]) < resolution:
        diff = resolution - (bbox[2] - bbox[0])
        bbox[0] -= diff // 2
        bbox[2] += diff // 2
    if (bbox[3] - bbox[1]) < resolution:
        diff = resolution - (bbox[3] - bbox[1])
        bbox[1] -= diff // 2
        bbox[3] += diff // 2
        
    return image.crop(bbox), mask.crop(bbox)

def is_blurry(pil_image: Image.Image) -> float:
    """Calculates a blurriness score using the variance of the Laplacian."""
    gray = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def image_to_base64(image: Image.Image, format="PNG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ----------------------------------------------------------------------
# ─── API ENDPOINT AND MODELS ──────────────────────────────────────────
# ----------------------------------------------------------------------

@app.post("/process/", response_model=ProcessedImagePayload)
async def process_image_endpoint(config: PreprocessingConfig):
    """Main unified image processing endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.image_url) as response:
                response.raise_for_status()
                image_bytes = await response.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or open image: {e}")

    # This logic block is now cleaner as it calls the imported module
    if config.use_bisenet and config.is_human_subject:
        print("🎭 Using BiseNet for face segmentation...")
        try:
            # The get_face_mask method returns multiple values; we only need the first (the mask)
            full_mask_np, _, _ = face_masker.get_face_mask(
                image,
                include_neck=True,
                include_hair=True,
            )
            mask = Image.fromarray(full_mask_np, 'L')
        except ValueError as e:
            # Fallback to BiRefNet if BiseNet fails to find a face
            print(f"⚠️ BiseNet Error: {e}. Falling back to BiRefNet.")
            mask = get_birefnet_mask(image)
    else:
        print("🎭 Using BiRefNet for face segmentation...")
        # Default to the existing BiRefNet method
        mask = get_birefnet_mask(image)

    image_cropped, mask_cropped = crop_to_square(image, mask, config.resolution, config.crop_expansion_factor)
    
    if config.black_mask:
        image_cropped = Image.composite(image_cropped, Image.new('RGB', image_cropped.size, (0,0,0)), mask_cropped)

    blur_factor, bbox, orig_bbox = None, None, None
    output_images = {}

    if config.face_crop and config.is_human_subject:
        bbox, orig_bbox = get_face_bbox(image_cropped, config.resolution)
        if bbox and orig_bbox:
            face_for_blur_check = ImageOps.fit(image_cropped.crop(orig_bbox), (512, 512))
            blur_factor = is_blurry(face_for_blur_check)
    
    if config.face_crop and config.is_human_subject and not bbox:
        return ProcessedImagePayload(blur_factor=None, skipped=True, images={})

    # Create "face" or "center-crop" image
    face_crop_img = ImageOps.fit(image_cropped.crop(bbox), (config.resolution, config.resolution)) if bbox else ImageOps.fit(image_cropped, (config.resolution, config.resolution))
    output_images["face_crop"] = image_to_base64(face_crop_img)

    # Create "padded" or "full-body" image
    padded_img = ImageOps.pad(image_cropped, (config.resolution, config.resolution))
    output_images["padded_image"] = image_to_base64(padded_img)

    # Create face mask
    if bbox and config.only_face and config.is_human_subject:
        isolated_face_mask = Image.new('L', mask_cropped.size)
        isolated_face_mask.paste(mask_cropped.crop(orig_bbox), (int(orig_bbox[0]), int(orig_bbox[1])))
        face_mask = ImageOps.fit(isolated_face_mask.crop(bbox), (config.resolution, config.resolution))
    else:
        face_mask = ImageOps.fit(mask_cropped, (config.resolution, config.resolution))
    output_images["face_mask"] = image_to_base64(face_mask)

    # Create padded mask
    padded_mask = ImageOps.pad(mask_cropped, (config.resolution, config.resolution))
    output_images["padded_mask"] = image_to_base64(padded_mask)
    
    return ProcessedImagePayload(blur_factor=blur_factor, skipped=False, images=output_images)

# if __name__ == "__main__":
#     print("Starting FastAPI server...")
#     print("Run with Gunicorn for production: gunicorn processor_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000")
#     uvicorn.run(app, host="0.0.0.0", port=10398)