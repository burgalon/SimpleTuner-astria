import os
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO


T = TypeVar("T", int, float)

MODELS_DIR = "/home/user/storage/astria/data/models"
CACHE_DIR = "/home/user/storage/astria/data/cache"
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

Bbox = tuple[float, float, float, float]

