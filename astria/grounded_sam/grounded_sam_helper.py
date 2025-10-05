import os

import numpy as np
from typing import Tuple
import re
import torch
from PIL import Image, ImageChops, ImageFilter

from groundingdino.util.inference import annotate, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import box_ops
from segment_anything.utils.transforms import ResizeLongestSide

from huggingface_hub import hf_hub_download
from astria_utils import device, MODELS_DIR, CACHE_DIR


SAM_REPO_ID = "HCMUE-Research/SAM-vit-h"
SAM_FILENAME = "sam_vit_h_4b8939.pth"
SAM_LOCAL_PATH = f"{CACHE_DIR}/{SAM_FILENAME}"


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def ensure_sam_checkpoint(
    repo_id: str = SAM_REPO_ID,
    filename: str = SAM_FILENAME,
    local_path: str = SAM_LOCAL_PATH,
    use_symlinks: bool = False,
) -> str:
    """
    Ensure the SAM checkpoint exists at `local_path`.
    If missing, download from HF and place/copy it there.

    Returns:
        str: absolute path to the checkpoint file (local_path).
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # If already there and non-empty, we're done.
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    # Try to download directly into desired directory (best path).
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=use_symlinks,  # False => real file copy
        )
        # hf_hub_download returns the absolute path to the file in local_dir.
        # It should already be at `local_path`, but guard just in case:
        if downloaded != local_path and os.path.exists(downloaded):
            try:
                # If user asked for a real file but hub made a symlink, or the
                # path differs, copy it to the exact target path.
                shutil.copy2(downloaded, local_path)
            except Exception:
                pass
        return local_path
    except Exception:
        # Fallback: download to HF cache, then copy to target.
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached, local_path)
        return local_path

# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )
  boxes = boxes[logits.argsort(descending=True)]
  phrases = [phrases[i] for i in logits.argsort(descending=True)]
  logits = logits[logits.argsort(descending=True)]

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB
  return annotated_frame, boxes

def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def transform_image(sam_predictor: "SamPredictor", image_source: Image) -> Tuple[np.array, torch.Tensor]:
    image = np.asarray(image_source)
    transform = ResizeLongestSide(sam_predictor.model.image_encoder.img_size)
    image_transformed = transform.apply_image(image)
    image_transformed = torch.as_tensor(image_transformed, device=device)
    print("transformed image shape", image_transformed.shape, image_source.size)
    return image, image_transformed.permute(2, 0, 1).contiguous().float() / 255.0

# https://github.com/schananas/grounded_sam_replicate/blob/main/grounded_sam.py#L77
def get_masks_by_class(input_image: Image, text_prompt, negative_prompts, class_index, groundingdino_model, sam_predictor, max_box_percent=None, box_threshold = 0.3, text_threshold = 0.25):
    image_source, image = transform_image(sam_predictor, input_image)

    # Commented for now because segmenting each person separately seems more important in cases where poses overlap each other
    if text_prompt=='foreground':
        raise Exception("Modnet not supported yet")
    else:
        annotated_frame, detected_boxes = detect(image, image_source, text_prompt=text_prompt, model=groundingdino_model, box_threshold=box_threshold, text_threshold=text_threshold)
        if os.environ.get('DEBUG') == 'mask':
            Image.fromarray(annotated_frame).save(f"{MODELS_DIR}/annotated_frame.jpg")
        if class_index is not None:
            print(f"DETECTED BOXES {len(detected_boxes)} class_index={class_index}")
            detected_boxes = detected_boxes[class_index:class_index+1]
        if max_box_percent is not None:
            # filter out boxes that are too small - boxes are cxcywh
            box_areas = detected_boxes[:, 2] * detected_boxes[:, 3]
            if os.environ.get('DEBUG') == 'mask': print(f"num boxes before filtering: {detected_boxes.shape[0]} - box_areas={box_areas}")
            detected_boxes = detected_boxes[(box_areas < max_box_percent)]
            if os.environ.get('DEBUG') == 'mask': print(f"num boxes after filtering: {detected_boxes.shape[0]}")

        if detected_boxes is None or detected_boxes.shape[0] == 0:
            return None
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
        merged_mask = np.logical_or.reduce(segmented_frame_masks[:, 0])

        # Converting positive mask into PIL image
        mask = (merged_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition


    if negative_prompts:
        for negative_prompt in negative_prompts:
            neg_annotated_frame, neg_detected_boxes = detect(image, image_source, text_prompt=negative_prompt, model=groundingdino_model)
            if neg_detected_boxes is None or len(neg_detected_boxes) == 0:
                continue
            neg_segmented_frame_masks = segment(image_source, sam_predictor, boxes=neg_detected_boxes)

            # Merging all negative masks
            merged_neg_mask = np.logical_or.reduce(neg_segmented_frame_masks[:, 0])

            neg_mask = (merged_neg_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition

            # Use logical operations to subtract the negative mask from the original mask
            mask = mask & ~neg_mask

    return Image.fromarray(mask)

if __name__ == "__main__":
    import sys
    from sam_helper import GROUNDING_DINO_MAPPING, SAM_CHECKPOINT_MAPPING
    from segment_anything import SamPredictor, build_sam
    from diffusers.utils import load_image

    image = load_image(sys.argv[1])
    prompt_text = sys.argv[2]
    negative_prompt = sys.argv[3] if len(sys.argv) > 3 else None
    class_index = int(sys.argv[4]) if len(sys.argv) > 4 else None
    ckpt_repo_id = GROUNDING_DINO_MAPPING["model"]
    ckpt_filename = GROUNDING_DINO_MAPPING["checkpoint"]
    ckpt_config_filename = GROUNDING_DINO_MAPPING["config"]
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
    sam_predictor = SamPredictor(build_sam(checkpoint=SAM_LOCAL_PATH).to(device))
    mask = get_masks_by_class(
        image,
        prompt_text,
        negative_prompt,
        class_index,
        groundingdino_model,
        sam_predictor
    )
    mask.save(sys.argv[1].split('.')[0] + '_mask.jpg')
