# general_inpaint.py
import cv2
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

import torch
from astria_utils import JsonObj, MODELS_DIR, device
from pipeline_flux_differential_img2img import FluxDifferentialImg2ImgPipeline

DEBUG_ENABLED = os.environ.get("MOCK_SERVER") == "1"

# Debug save directory
DEBUG_SAVE_DIR = "/data/models"


@dataclass
class RegionSession:
    active: bool
    # originals
    original_image: Image.Image | None = None
    # pre-call artifacts
    pre_image: Image.Image | None = None
    pre_mask_for_pipe: Image.Image | None = None  # SOFT mask (inverted if needed by pipeline)
    pre_blend_mask: Image.Image | None = None     # SOFT outward blend for paste-back
    crop_box: tuple[int,int,int,int] | None = None
    in_h: int | None = None
    in_w: int | None = None
    prev_image_in_kwargs: Image.Image | None = None
    prev_mask_in_kwargs: Image.Image | None = None
    prev_strength_in_kwargs: float | None = None


# ---------- Reuse/bring-over a few helpers from your inpaint_faces mixin ----------
def _ensure_binary_u8(mask: Image.Image) -> np.ndarray:
    """Return binary mask (0/255) uint8."""
    m = np.array(mask.convert("L"), dtype=np.uint8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

# --- robust converters ---
def _to_int_default(val, default):
    try:
        # accept floats, strings like "64", etc.
        v = int(val)
    except (TypeError, ValueError):
        return default
    return v

def _nonneg_int(val, default):
    v = _to_int_default(val, default)
    return max(0, v)

def _pos_int(val, default):
    v = _to_int_default(val, default)
    return max(1, v)

# --- existing helpers but made None-safe ---
def _dilate(mask_np: np.ndarray, radius_px: int | None) -> np.ndarray:
    r = 0 if radius_px is None else int(max(0, radius_px))
    if r == 0:
        return mask_np
    k = r if r % 2 == 1 else r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask_np, kernel, iterations=1)


def _block_reduce_max(mask_np: np.ndarray, cell_w: int | None, cell_h: int | None) -> np.ndarray:
    H, W = mask_np.shape
    cw = int(cell_w or 64)
    ch = int(cell_h or 64)
    cw = max(1, cw); ch = max(1, ch)

    pad_w = (cw - (W % cw)) % cw
    pad_h = (ch - (H % ch)) % ch
    if pad_w or pad_h:
        mask_np = cv2.copyMakeBorder(mask_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    H2, W2 = mask_np.shape

    grid_h = H2 // ch
    grid_w = W2 // cw
    blocks = mask_np.reshape(grid_h, ch, grid_w, cw)
    reduced = blocks.max(axis=(1, 3))  # (grid_h, grid_w)

    up = np.kron(reduced, np.ones((ch, cw), dtype=np.uint8))
    return up[:H, :W]


def make_coarse_binary_mask(
    mask: Image.Image,
    *,
    expand_px: int | None = 48,
    grid_cell_px: int | None = 64,
) -> Image.Image:
    """
    Produce a chunky, **binary** (0/255) mask:
      1) morphological dilate by `expand_px`
      2) grid blockify with `grid_cell_px`
      3) re-binarize to 0/255 (no blurry values)
    """
    base = _ensure_binary_u8(mask)
    grown = _dilate(base, expand_px)
    coarse = _block_reduce_max(grown, grid_cell_px, grid_cell_px)
    # ensure strictly binary
    _, binary = cv2.threshold(coarse, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary, mode="L")


def make_soft_inpaint_mask_from_binary(
    binary_mask: Image.Image,
    *,
    outer_feather_px: int = 128,
) -> Image.Image:
    """
    From a **binary** (0/255) mask, create a **soft** inpaint mask (0..255)
    that is 255 inside and only feathers outward into 0 outside.
    This avoids any inner gradient, so inverting the mask will NOT produce an
    inner bright outline.
    """
    m = _ensure_binary_u8(binary_mask)
    if outer_feather_px <= 0:
        return Image.fromarray(m, mode="L")

    inv = cv2.bitwise_not(m)
    dist_out = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # distance outside

    # start from solid inside (255), fall off outside
    soft = m.astype(np.float32)
    outside = (m == 0)
    soft[outside] = np.clip(
        255.0 - (dist_out[outside] / max(1e-6, outer_feather_px)) * 255.0,
        0.0, 255.0
    )
    return Image.fromarray(soft.astype(np.uint8), mode="L")


def make_outward_blend_from_binary(
    binary_mask: Image.Image,
    *,
    feather_out_px: int = 128,
) -> Image.Image:
    """
    Optional: create a **grayscale blend mask** for final paste-back, based on a **binary** mask.
    Keeps inside at 255 and only feathers into the outside.
    """
    if feather_out_px <= 0:
        return binary_mask.convert("L")
    m = _ensure_binary_u8(binary_mask)
    inv = cv2.bitwise_not(m)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    alpha_out = 255 - (dist / max(1e-6, feather_out_px)) * 255
    alpha_out = np.clip(alpha_out, 0, 255).astype(np.uint8)
    out = m.copy()
    outside = (m == 0)
    out[outside] = alpha_out[outside]
    return Image.fromarray(out, mode="L")


def _feather_outwards(mask_np: np.ndarray, radius_px: int | None) -> np.ndarray:
    """Keep inside 255; outward falloff only. None/0 => return input."""
    r = _nonneg_int(radius_px, 0)
    if r == 0:
        return mask_np
    inv = cv2.bitwise_not(mask_np)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    alpha_out = 255 - (dist / max(1e-6, r)) * 255
    alpha_out = np.clip(alpha_out, 0, 255).astype(np.uint8)
    out = mask_np.copy()
    outside = (mask_np == 0)
    out[outside] = alpha_out[outside]
    return out


def make_coarse_inpaint_mask(
    mask: Image.Image,
    *,
    expand_px: int = 48,          # outward grow (≈ 0.75 * 64 works well)
    grid_cell_px: int = 64,       # size of square grid cells
    feather_out_px: int = 2*64,   # outward feather amount for pasting
    blockify_first: bool = True,  # usually True: grow, blockify, then feather
) -> Image.Image:
    """
    Turn a fine mask into a diffusion-friendly chunky mask.

    Steps:
      1) Dilate by `expand_px` to include helpful context.
      2) Blockify into `grid_cell_px` tiles using max-pooling and NEAREST upsample.
      3) Feather outward (optional) for nicer composite edges.

    Returns: PIL 'L' image (0..255).
    """
    base = _ensure_binary_u8(mask)

    # 1) Grow
    grown = _dilate(base, expand_px)

    # 2) Blockify
    if blockify_first:
        coarse = _block_reduce_max(grown, grid_cell_px, grid_cell_px)
    else:
        coarse = grown

    # 3) Feather outward (keeps inside 255)
    coarse_feather = _feather_outwards(coarse, feather_out_px)

    return Image.fromarray(coarse_feather, mode="L")


def bbox_area(b: Tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def clamp_bbox_to_image(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mask_to_bbox(mask: Image.Image, min_area: int = 64) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox around non-zero mask; None if no foreground."""
    arr = np.array(mask.convert("L"))
    ys, xs = np.where(arr > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    b = (x1, y1, x2, y2)
    return b if bbox_area(b) >= min_area else None


def round_to_multiple_of(v: int, m: int) -> int:
    return int(round(v / m) * m)


def round_up_to_multiple_of(v, m): return int((v + m - 1) // m * m)


def ensure_divisible_32(w: int, h: int):
    return (
        max(32, round_up_to_multiple_of(w, 32)),
        max(32, round_up_to_multiple_of(h, 32)),
    )

def pad_bbox(b: Tuple[int, int, int, int], pad_px: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return clamp_bbox_to_image((x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px), W, H)

# --- Feathering & color restore (ported) ---

def feather_mask_sdf(mask_pil: Image.Image, feather_radius: int = 200, upscale_factor: int = 4) -> Image.Image:
    import cv2
    width, height = mask_pil.size
    upw, uph = width * upscale_factor, height * upscale_factor
    large = mask_pil.resize((upw, uph), Image.NEAREST)
    mask_np = np.array(large).astype(np.uint8)
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    dist_out = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sdf = dist_out - dist_in
    alpha = np.zeros_like(sdf, dtype=np.float32)
    alpha[sdf <= -feather_radius] = 255
    alpha[sdf >= feather_radius] = 0
    zone = (sdf > -feather_radius) & (sdf < feather_radius)
    alpha[zone] = (1 - ((sdf[zone] + feather_radius) / (2 * feather_radius))) * 255
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha, mode='L').resize((width, height), Image.LANCZOS)


def feather_mask_outwards(mask_pil: Image.Image, feather_radius: int = 400, upscale_factor: int = 4) -> Image.Image:
    import cv2
    width, height = mask_pil.size
    upw, uph = width * upscale_factor, height * upscale_factor
    large = mask_pil.resize((upw, uph), Image.NEAREST)
    mask_np = np.array(large).astype(np.uint8)
    _, binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 5)
    alpha = np.ones_like(dist, dtype=np.float32) * 255
    alpha_out = 255 - (dist / max(1e-6, feather_radius)) * 255
    alpha_out = np.clip(alpha_out, 0, 255)
    outside = (binary == 0)
    alpha[outside] = alpha_out[outside]
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha, mode='L').resize((width, height), Image.LANCZOS)


def restore_colors(original: Image.Image, inpainted: Image.Image) -> Image.Image:
    from skimage.exposure import match_histograms
    A = np.array(original)
    B = np.array(inpainted)
    matched = match_histograms(B, A, channel_axis=-1)
    return Image.fromarray(matched)

# ---------- Grounded SAM adapter (only used when text-prompted mask is needed) ----------

class GroundedSamAdapter:
    """
    Lazily loads GroundingDINO + SAM and provides `make_mask(image, text_prompt, negatives=None, class_index=None, ...)`.
    Plug your existing helper from the prompt into this wrapper to keep imports tidy.
    """
    def __init__(self, loader_fn):
        """
        loader_fn: callable -> (groundingdino_model, sam_predictor)
        """
        self.loader_fn = loader_fn
        self._loaded = False
        self.gd = None
        self.sam = None

    def _ensure(self):
        if not self._loaded:
            self.gd, self.sam = self.loader_fn()
            self._loaded = True

    def make_mask(
        self,
        input_image: Image.Image,
        text_prompt: str,
        negative_prompts: Optional[List[str]] = None,
        class_index: Optional[int] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        max_box_percent: Optional[float] = None,
    ) -> Optional[Image.Image]:
        """
        Returns a binary 'L' mask (0/255) or None.
        """
        from astria.grounded_sam.grounded_sam_helper import get_masks_by_class
        self._ensure()
        mask = get_masks_by_class(
            input_image,
            text_prompt,
            negative_prompts,
            class_index,
            self.gd,
            self.sam,
            max_box_percent=max_box_percent,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return mask

# ---------- General-purpose region inpaint mixin ----------

INPAINT_TARGET_MP = 1.0  # ~1 megapixel
INPAINT_TARGET_EDGE = 1024  # prefer 1024x1024 when aspect ~1
MIN_EDGE = 256
MAX_EDGE = 2048  # safety clamp; adjust based on VRAM
DIVISOR = 64


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    # Scale, bias and saturate x to 0..1
    if edge1 == edge0:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def compute_region_inpaint_strength(bbox_ratio: float, override: float | None = None) -> float:
    """
    Gentler heuristic for img2img inpaint strength.
    - Small regions (<=0.5%) → ~0.62
    - Medium regions → ~0.55
    - Large regions (>=20%) → ~0.48
    """
    if override is not None:
        return float(override)

    R_LOW, R_HIGH = 0.005, 0.20   # ratio band to interpolate over
    S_MIN, S_MAX = 0.48, 0.62     # allowed strength range (narrow, less extreme)

    t = _smoothstep(bbox_ratio, R_LOW, R_HIGH)  # 0 at small, 1 at large
    strength = S_MAX - (S_MAX - S_MIN) * t      # small→S_MAX, large→S_MIN
    return float(max(S_MIN, min(S_MAX, strength)))


class InpaintRegionMixin:
    """
    A general-purpose inpainting helper:
      1) obtains a mask via prompt.mask_image OR Grounded-SAM from prompt.mask_prompt
      2) chooses a crop canvas that contains the target bbox and ~1MP area (multiples of 32)
      3) runs the inpaint pipeline on the cropped region
      4) feather-blends the result back into the original frame

    Dependencies:
      - self.init_inpaint(prompt) -> sets self.inpaint or self.fill pipeline (like in your faces mixin)
      - optional SR: self.sr_model, load_sr(), upscale_sr() if you want pre/post upscaling (same as faces)
    """

    def __init__(self):
        self.grounded_sam: Optional[GroundedSamAdapter] = None
        self.sr_model = None  # if you plan to reuse SR

    # --------- public API ---------

    def inpaint_regions(self, images: List[Image.Image], prompt: JsonObj, kwargs: Dict) -> List[Image.Image]:
        """
        High-level entry point. Works image-by-image.
        Required prompt fields (one of):
          - prompt.mask_image (PIL.Image in 'L' or '1'), OR
          - prompt.mask_prompt (str) + Grounded-SAM available via self.get_grounded_sam_loader()

        Optional:
          - prompt.bbox: [x1,y1,x2,y2] to force the region
          - prompt.mask_negative_prompts: List[str]
          - prompt.class_index / prompt.box_threshold / prompt.text_threshold / prompt.max_box_percent
          - prompt.super_resolution: bool
          - prompt.region_inpaint_strength: float  (fallback heuristic if None)
        """
        self.init_inpaint(prompt)  # from your existing pipeline initializer
        out = []
        debug_enabled = DEBUG_ENABLED
        
        for i, im in enumerate(images):
            # Save original image for debugging
            if debug_enabled:
                os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
                im.save(f"{DEBUG_SAVE_DIR}/{prompt.id}-000-original-{i}.png")
                print(f"[DEBUG] Saved original image to {DEBUG_SAVE_DIR}/{prompt.id}-000-original-{i}.png")
            
            try:
                out_im = self._inpaint_single_region(im, prompt, kwargs, image_idx=i)
            except Exception as e:
                print(f"Region inpaint failed on image {i}: {e}")
                out_im = im
                raise e
            out.append(out_im)
        return out

   # public tiny helper: allow `--mask_crop` to snap bbox to mask
    def crop_mask(self, prompt):
        m = getattr(prompt, "mask_image", None)
        if isinstance(m, Image.Image):
            b = mask_to_bbox(m)
            if b:
                prompt.bbox = list(b)  # used by the session builder

    def maybe_build_region_session(
        self,
        prompt,
        pipe,
        kwargs,
    ) -> RegionSession:
        """
        If user signaled a region (mask_image or mask_prompt) and this is an img2img/inpaint
        scenario, build a RegionSession; otherwise return an inactive session.
        Does NOT change the control flow; you still call `pipe(...)` as usual.
        """
        # do not fight ControlNet flows unless you want to allow it
        if getattr(prompt, "controlnet", None):
            return RegionSession(active=False)

        # require an img2img context
        base_image = kwargs.get("image", None)
        if base_image is None or not isinstance(base_image, Image.Image):
            return RegionSession(active=False)

        # resolve / prepare a mask if needed
        if getattr(prompt, "mask_image", None) is None and getattr(prompt, "mask_prompt", None):
            # re-use your existing fine-mask helper from the mixin
            prompt.mask_image = self._obtain_mask_region(base_image, prompt)

        if getattr(prompt, "mask_crop", False):
            self.crop_mask(prompt)

        # if still no mask -> nothing to do
        if getattr(prompt, "mask_image", None) is None:
            return RegionSession(active=False)

        # ----- build the same artifacts as _inpaint_single_region, but stop before calling pipe
        W, H = base_image.size
        fine_mask = prompt.mask_image.convert("L")

        # bbox (from user bbox or mask)
        if getattr(prompt, "bbox", None):
            bbox = clamp_bbox_to_image(tuple(map(int, prompt.bbox)), W, H)
        else:
            d = mask_to_bbox(fine_mask)
            if not d:
                return RegionSession(active=False)
            bbox = d

        # margin (auto vs manual mask)
        is_auto = bool(getattr(prompt, "mask_prompt", None))
        margin_pct = 0.15 if is_auto else 0.10
        margin = int(margin_pct * max(bbox[2]-bbox[0], bbox[3]-bbox[1]))
        bbox = pad_bbox(bbox, margin, W, H)

        # choose crop canvas
        crop_box, (crop_w, crop_h) = self._choose_crop_canvas_region(bbox, (W, H))
        cx1, cy1, cx2, cy2 = crop_box

        # crop
        cropped_img = base_image.crop(crop_box)
        cropped_mask_fine = fine_mask.crop(crop_box)

        # coarsen binary if auto; otherwise just binarize
        if is_auto:
            grid_px   = int(getattr(prompt, "mask_grid_px", 16) or 16)
            expand_px = int(getattr(prompt, "mask_expand_px", 16) or 16)
            cropped_mask_bin = make_coarse_binary_mask(
                cropped_mask_fine, expand_px=expand_px, grid_cell_px=grid_px
            )
        else:
            cropped_mask_bin = Image.fromarray(_ensure_binary_u8(cropped_mask_fine), mode="L")

        # soft outward mask for diffusion & blending
        soft_out_px = int(getattr(prompt, "inpaint_mask_feather_px", 128) or 128)
        cropped_mask_soft = make_soft_inpaint_mask_from_binary(
            cropped_mask_bin, outer_feather_px=soft_out_px
        )

        # resize to /32
        in_w, in_h = ensure_divisible_32(crop_w, crop_h)
        if (in_w, in_h) != (crop_w, crop_h):
            cropped_img       = cropped_img.resize((in_w, in_h), Image.LANCZOS)
            cropped_mask_soft = cropped_mask_soft.resize((in_w, in_h), Image.BILINEAR)

        # per-region strength heuristic (gentle); do not override global unless user asked
        bbox_ratio = bbox_area(bbox) / float(W * H)
        default_strength = compute_region_inpaint_strength(
            bbox_ratio, getattr(prompt, "region_inpaint_strength", None)
        )

        # mask convention for FluxDifferential
        mask_for_pipe = cropped_mask_soft
        from pipeline_flux_differential_img2img import FluxDifferentialImg2ImgPipeline
        if isinstance(pipe, FluxDifferentialImg2ImgPipeline):
            mask_for_pipe = ImageOps.invert(mask_for_pipe)

        # preserve previous kwargs to restore later if needed
        sess = RegionSession(
            active=True,
            original_image=base_image,
            pre_image=cropped_img,
            pre_mask_for_pipe=mask_for_pipe,
            pre_blend_mask=cropped_mask_soft,  # same soft mask, non-inverted, for paste-back
            crop_box=crop_box,
            in_h=in_h,
            in_w=in_w,
            prev_image_in_kwargs=kwargs.get("image"),
            prev_mask_in_kwargs=kwargs.get("mask_image"),
            prev_strength_in_kwargs=kwargs.get("strength"),
        )

        # mutate kwargs minimally for the normal pipe call
        kwargs["image"] = sess.pre_image
        kwargs["mask_image"] = sess.pre_mask_for_pipe
        # only set strength if user didn't provide one explicitly
        if kwargs.get("strength", None) is None:
            kwargs["strength"] = float(default_strength)

        # IMPORTANT: do not touch guidance / embeds; infer_prompt already set them
        return sess

    def post_region_paste(self, sess: RegionSession, img_out: Image.Image) -> Image.Image:
        """
        After your normal `pipe(...)` call returns an image, call this to paste the
        result back into the original canvas. No-ops if session is inactive.
        """
        if not sess or not sess.active:
            return img_out

        cx1, cy1, cx2, cy2 = sess.crop_box
        W = sess.original_image.width
        H = sess.original_image.height

        # resize output + blend mask to crop size (safety)
        out = img_out
        if out.size != (cx2 - cx1, cy2 - cy1):
            out = out.resize((cx2 - cx1, cy2 - cy1), Image.LANCZOS)
        blend = sess.pre_blend_mask
        if blend.size != (cx2 - cx1, cy2 - cy1):
            blend = blend.resize((cx2 - cx1, cy2 - cy1), Image.BILINEAR)

        base = sess.original_image
        comp = base.copy()
        region = Image.composite(out, base.crop((cx1, cy1, cx2, cy2)), blend)
        comp.paste(region, (cx1, cy1))
        return comp

    # --------- internal helpers ---------
    def _inpaint_single_region(self, image: Image.Image, prompt: JsonObj, kwargs: Dict, image_idx: int = 0) -> Image.Image:
        """
        Now a thin wrapper that reuses the session helpers:
        - maybe_build_region_session(...) prepares aligned crop + soft mask (+ strength)
        - pipe(...) runs at crop resolution
        - optional color match on the crop
        - post_region_paste(...) blends back into full canvas
        """
        debug_enabled = DEBUG_ENABLED

        # we use the same inpaint pipe you already initialize in inpaint_regions()
        pipe = self.inpaint or self.fill

        # Build a LOCAL kwargs view that points to `image` so the session logic can align
        # everything relative to this frame. This avoids mutating the caller's kwargs unexpectedly.
        local_kwargs = dict(kwargs)
        local_kwargs["image"] = image

        # Let the unified builder do: bbox, crop, soft mask, /32 sizing, default strength, inversion for FluxDifferential
        sess = self.maybe_build_region_session(prompt, pipe, local_kwargs)
        if not sess or not sess.active:
            # no region (no mask / controlnet active / invalid bbox) -> keep original
            if debug_enabled:
                print(f"T#{prompt.tune_id} P#{prompt.id} region_inpaint: inactive session; returning original")
            return image

        in_w, in_h = sess.in_w, sess.in_h
        cropped_img = sess.pre_image
        mask_for_pipe = sess.pre_mask_for_pipe

        if debug_enabled:
            os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
            # Save diagnostics if you like; these are already produced by the session
            # but it's harmless to log here:
            cropped_img.save(f"{DEBUG_SAVE_DIR}/{prompt.id}-001-cropped-image-{image_idx}.png")

        # Carry over only prompt/guidance bits; your code already does this pattern
        pipe_overrides = {
            k: v for k, v in kwargs.items()
            if ('prompt' in k) or (k in ('true_cfg_scale', 'guidance_scale'))
            if v is not None
        }
        pipe_overrides.setdefault('guidance_scale', 3.5)

        # strength: session already wrote a default into local_kwargs if caller didn't provide one
        strength = float(local_kwargs.get("strength"))

        if hasattr(pipe.transformer, 'clear_cache'):
            pipe.transformer.clear_cache()

        if debug_enabled:
            print(f"[DEBUG] Running inpaint pipeline at {in_w}x{in_h} (strength={strength:.3f})...")

        # Run the pipeline at crop resolution (exactly as before, just using session-prepped inputs)
        inpainted = pipe(
            height=in_h,
            width=in_w,
            num_inference_steps=28,
            generator=torch.Generator(device=device).manual_seed(42),
            **({} if 'Qwen' in pipe.__class__.__name__ else {"joint_attention_kwargs": {"scale": 1.0}}),
            mask_image=mask_for_pipe,
            image=cropped_img,
            strength=strength,
            **pipe_overrides,
        ).images[0]

        inpainted.save(f"{DEBUG_SAVE_DIR}/{prompt.id}-002-inpainted-raw-{image_idx}.png")

        # 🔧 Paste the crop result back into the original canvas so output size == input size
        composited = self.post_region_paste(sess, inpainted)
        if debug_enabled:
            composited.save(f"{DEBUG_SAVE_DIR}/{prompt.id}-003-composited-{image_idx}.png")
        return composited

    # ---------- mask acquisition ----------

    def _obtain_mask_region(self, image: Image.Image, prompt: JsonObj) -> Optional[Image.Image]:
        # 1) If mask image provided, use it directly (accept RGB/LA/etc; convert to L)
        if getattr(prompt, "mask_image", None) is not None:
            m = prompt.mask_image
            if not isinstance(m, Image.Image):
                raise TypeError("prompt.mask_image must be a PIL.Image")
            return m.convert("L")

        # 2) If text prompt given, auto-detect with Grounded-SAM
        if getattr(prompt, "mask_prompt", None):
            adapter = self._get_gsam()
            negs = getattr(prompt, "mask_negative_prompts", None)
            cls_idx = getattr(prompt, "class_index", None)
            box_thr = 0.3 if getattr(prompt, "box_threshold", None) is None else float(prompt.box_threshold)
            txt_thr = 0.25 if getattr(prompt, "text_threshold", None) is None else float(prompt.text_threshold)
            max_box_percent = getattr(prompt, "max_box_percent", None)
            mask = adapter.make_mask(
                image,
                prompt.mask_prompt,
                negative_prompts=negs,
                class_index=cls_idx,
                box_threshold=box_thr,
                text_threshold=txt_thr,
                max_box_percent=max_box_percent,
            )
            return mask

        # No mask info available
        return None

    def get_grounded_sam_loader(self):
        # Return a function that loads and returns (groundingdino_model, sam_predictor)
        # You already have this in your snippet; just wrap it.
        from sam_helper import GROUNDING_DINO_MAPPING, SAM_CHECKPOINT_MAPPING
        from segment_anything import SamPredictor, build_sam
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        from huggingface_hub import hf_hub_download
        from astria.grounded_sam.grounded_sam_helper import ensure_sam_checkpoint
        import torch

        def _loader():
            repo = GROUNDING_DINO_MAPPING["model"]
            ckpt = GROUNDING_DINO_MAPPING["checkpoint"]
            cfg  = GROUNDING_DINO_MAPPING["config"]
            cfg_path = hf_hub_download(repo_id=repo, filename=cfg)
            args = SLConfig.fromfile(cfg_path); args.device = device
            gd = build_model(args)
            ckpt_path = hf_hub_download(repo_id=repo, filename=ckpt)
            sd = torch.load(ckpt_path, map_location=device)['model']
            _ = gd.load_state_dict(clean_state_dict(sd), strict=False)
            gd.eval()

            sam_path = ensure_sam_checkpoint()
            sam = SamPredictor(build_sam(checkpoint=sam_path).to(device))
            return gd, sam

        return _loader

    def _get_gsam(self) -> GroundedSamAdapter:
        """
        Override or provide loader that returns (groundingdino_model, sam_predictor).
        By default, looks for a method self.get_grounded_sam_loader() you can implement
        (so you can reuse the code you pasted in your message).
        """
        if self.grounded_sam is None:
            if not hasattr(self, "get_grounded_sam_loader"):
                raise RuntimeError("Grounded-SAM is required but no loader found. Implement self.get_grounded_sam_loader().")
            self.grounded_sam = GroundedSamAdapter(self.get_grounded_sam_loader())
        return self.grounded_sam

    # ---------- crop sizing logic ----------

    def _choose_crop_canvas_region(
        self,
        bbox: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        target_mp: float = INPAINT_TARGET_MP,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        """
        Picks a crop box (x1,y1,x2,y2) that fully contains `bbox` while trying to get area ~ target_mp megapixels.
        Keeps aspect ~ bbox's aspect, clamps to image, and rounds to multiples of 32.
        """
        W, H = img_size
        bx1, by1, bx2, by2 = bbox
        bw, bh = (bx2 - bx1), (by2 - by1)
        bw = max(1, bw); bh = max(1, bh)

        bbox_aspect = bw / bh
        target_area = target_mp * 1_000_000.0

        # initial desired size
        h_est = math.sqrt(target_area / bbox_aspect)
        w_est = h_est * bbox_aspect

        # clamp to image bounds and reasonable edges
        w_est = int(min(MAX_EDGE, max(MIN_EDGE, w_est)))
        h_est = int(min(MAX_EDGE, max(MIN_EDGE, h_est)))

        # must at least contain bbox
        w_need = max(w_est, bw)
        h_need = max(h_est, bh)

        # round to /32
        w_need, h_need = ensure_divisible_32(w_need, h_need)

        # center the bbox inside the canvas if possible
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        x1 = int(cx - w_need // 2)
        y1 = int(cy - h_need // 2)
        x2 = x1 + w_need
        y2 = y1 + h_need

        # clamp crop to image; if clamped, keep size but move box inside
        dx = 0; dy = 0
        if x1 < 0: dx = -x1
        if y1 < 0: dy = -y1
        if x2 > W: dx = min(dx, W - x2)
        if y2 > H: dy = min(dy, H - y2)
        x1 += dx; x2 += dx; y1 += dy; y2 += dy

        # final clamp to image (in case extremely near edges)
        x1, y1, x2, y2 = clamp_bbox_to_image((x1, y1, x2, y2), W, H)
        # Fix size if clamped changed dimensions: recompute to /32 while staying within image
        w_final = x2 - x1; h_final = y2 - y1
        w_final, h_final = ensure_divisible_32(w_final, h_final)
        # If rounding shrunk, adjust x2/y2
        x2 = min(W, x1 + w_final); y2 = min(H, y1 + h_final)
        # If we lost coverage on bbox, expand back if we can
        if not (x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2):
            # fallback: just take the minimal padded bbox to /32
            min_w = round_to_multiple_of(bw, DIVISOR)
            min_h = round_to_multiple_of(bh, DIVISOR)
            x1 = max(0, bx1 - (min_w - bw)//2); y1 = max(0, by1 - (min_h - bh)//2)
            x2 = min(W, x1 + min_w); y2 = min(H, y1 + min_h)
            w_final = x2 - x1; h_final = y2 - y1
            w_final, h_final = ensure_divisible_32(w_final, h_final)
            x2 = min(W, x1 + w_final); y2 = min(H, y1 + h_final)

        return (x1, y1, x2, y2), (x2 - x1, y2 - y1)
