import json
import os
import shutil
import time
import cv2

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from PIL import Image, ImageOps, ImageStat, ImageEnhance
from tqdm import tqdm
from ultralytics import YOLO

from astria_utils import run, run_with_output, EPHEMERAL_MODELS_DIR, JsonObj, device, HUMAN_CLASS_NAMES
from birefnet.BiRefNet_node import BiRefNet_node
from image_utils import io2img, save_img as save_img_base
from inpaint_face_mixin import ultralytics_predict, filter_by_ratio, filter_k_largest, YOLO_FACE_MODEL

import albumentations as A


# --- Resampling helpers --------------------------------------------------------
def fit_img(im, size):
    # CHANGED: use Lanczos (sharper when upscaling) instead of Bicubic
    return ImageOps.fit(im, size, method=Image.LANCZOS)

def fit_mask(m, size):
    return ImageOps.fit(m, size, method=Image.NEAREST)

def pad_img(im, size):
    # CHANGED: use Lanczos
    return ImageOps.pad(im, size, method=Image.LANCZOS)

def pad_mask(m, size):
    return ImageOps.pad(m, size, method=Image.NEAREST)


# Build one global augmenter (square output = `resolution × resolution`)
def build_augmenter(resolution: int) -> A.Compose:
    """
    CHANGED:
      - Remove RandomResizedCrop (it always resizes → blur when scale<1).
      - Keep small zooms via Affine(scale≈1).
      - No final Resize; we end with RandomCrop to get exact size.
      - Use LANCZOS4 for any upsampling that Affine performs.
    """
    return A.Compose(
        [
            A.Affine(
                scale=(0.90, 1.06),            # mild zoom in/out (keeps detail)
                keep_ratio=True,
                fit_output=False,              # avoid canvas resize
                interpolation=cv2.INTER_LANCZOS4,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.9,
            ),

            # Ensure we can always crop to target size without upscaling
            A.PadIfNeeded(
                min_height=resolution, min_width=resolution,
                border_mode=cv2.BORDER_REFLECT_101,
                value=None, mask_value=0,
                always_apply=True
            ),

            # Produce exact training size with no resampling
            A.RandomCrop(height=resolution, width=resolution, always_apply=True),

            # Photometrics
            A.CLAHE(clip_limit=2, p=0.5),
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.5),
                saturation=0.05,
                hue=0.0,
                p=0.8,
            ),

            # Keep mask binary
            A.Lambda(
                mask=lambda m, **k: (
                    ((m > 0.5) if m.dtype != np.uint8 else (m > 127))
                    .astype("uint8") * 255
                ),
                p=1.0,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def save_img_augs(
    img: Image.Image,
    path: str,
    *,
    mask: Image.Image | None = None,
    mask_dir: str | None = None,
    n_aug: int = 8,
    keep_original: bool = True,
    augmenter: A.Compose | None = None,
):
    """
    Save `img` (and optional `mask`) plus `n_aug` augmented variants.
    """
    from image_utils import save_img as _save_img_orig

    if augmenter is None:
        raise ValueError("Pass an Albumentations augmenter")

    base_dir, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)

    # --- 0) save the un-augmented image/mask -----------------
    if keep_original:
        save_img_base(img, path)
        if mask is not None:
            save_img_base(mask, os.path.join(mask_dir, filename))

    # --- 1) create N augments --------------------------------
    img_np  = np.array(img)
    mask_np = np.array(mask) if mask is not None else None

    for k in range(n_aug):
        out = augmenter(image=img_np, mask=mask_np)
        aug_img  = Image.fromarray(out["image"])
        aug_path = os.path.join(base_dir, f"{stem}_aug{k:02d}{ext}")
        save_img_base(aug_img, aug_path)

        if mask is not None:
            aug_mask = Image.fromarray(out["mask"])
            aug_mask_path = os.path.join(mask_dir, f"{stem}_aug{k:02d}{ext}")
            save_img_base(aug_mask, aug_mask_path)



def list_full_paths(directory: str) -> list[str]:
    return [os.path.join(os.path.abspath(directory), f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]


def infer_background(image: Image.Image, birefnet: BiRefNet_node) -> np.ndarray:
    alpha_tensor : torch.Tensor = birefnet.matting(image, 'cuda')
    alpha = alpha_tensor.squeeze().cpu().numpy()
    return alpha


def auto_adjust_contrast_brightness(
    img: Image.Image, 
    target_mean: float = 128.0,
    clip_percent: float = 0.0,
) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=clip_percent)
    img = ImageOps.equalize(img)
    stat = ImageStat.Stat(img.convert("L"))
    mean_lum = stat.mean[0]
    if mean_lum == 0:
        return img
    factor = float(target_mean) / mean_lum
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    return img


def enhance_luma_only(img: Image.Image, clip_percent=0):
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.autocontrast(y, cutoff=clip_percent)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def get_mask(image: Image.Image, birefnet: BiRefNet_node, mask_dir: str, image_name: str) -> None:
    alpha = infer_background(image, birefnet)
    mask = (alpha >= 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask)
    return mask_image

BASE_TRAIN_RESOLUTION = 512

# define Bbox
Bbox = tuple[float, float, float, float]

def get_face_bbox(image: Image.Image, yolo) -> (Bbox, Bbox):
    pred = ultralytics_predict(
        yolo,
        image=image,
        confidence=0.3,
        device=device,
        classes=None,
    )
    pred = filter_by_ratio(pred, low=0.003, high=1)
    pred = filter_k_largest(pred, k=0)
    pred.orig_bboxes = [[x1-(x2-x1)*0.2, y1-(y2-y1)*.2, x2+(x2-x1)*0.2, y2+(y2-y1)*.1] for x1, y1, x2, y2 in pred.bboxes]
    pred.bboxes = [[x1-(x2-x1)*0.4, y1-(y2-y1)*.5, x2+(x2-x1)*0.4, y2+(y2-y1)*.5] for x1, y1, x2, y2 in pred.bboxes]
    if len(pred.bboxes) == 0:
        return (None, None)

    bbox = pred.bboxes[0]
    orig_bbox = pred.orig_bboxes[0]
    if bbox[2] - bbox[0] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[2] - bbox[0])
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])
    if bbox[3] - bbox[1] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[3] - bbox[1])
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)

    return (bbox, orig_bbox) if bbox else (None, None)


def crop_mask_to_square(image: Image.Image, mask: Image.Image) -> [Image.Image, Image.Image]:
    bbox = mask.getbbox()
    if not bbox:
        print("crop_mask_to_square(): Failed to get bbox for mask")
        return image, mask
    bbox = list(bbox)
    bbox[1] = max(0, bbox[1] - (bbox[3] - bbox[1])*.1)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width > height:
        diff = width - height
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)
    elif height > width:
        diff = height - width
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])
    if bbox[2] - bbox[0] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[2] - bbox[0])
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])
    if bbox[3] - bbox[1] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[3] - bbox[1])
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)
    return image.crop(bbox), mask.crop(bbox)

@dataclass
class DownloadTrainingOutput():
    training_dir: str
    face_dir: str
    mask_dir: str
    resolution: int
    has_caption: bool
    all_captioned: bool


def is_blurry(pil_image: Image.Image) -> float:
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum.mean()

def is_slr(image: Image.Image) -> bool:
    try:
        for key, value in image.getexif().items():
            if isinstance(value, str) and ('Canon' in value or 'Nikon' in value):
                return True
        return False
    except SyntaxError:
        return False

def download_training(tune: JsonObj, one_dir=False):
    global BASE_TRAIN_RESOLUTION
    resolution = int(tune.resolution) if tune.resolution else BASE_TRAIN_RESOLUTION
    has_captions = False
    all_captioned = False
    BASE_TRAIN_RESOLUTION = resolution
    if one_dir:
        training_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
        mask_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
        face_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
    else:
        training_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
        mask_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-masks"
        face_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-faces"

    if os.environ.get('SKIP_DOWNLOAD'):
        print("Skipping download as SKIP_DOWNLOAD is set")
        return DownloadTrainingOutput(training_dir, face_dir, mask_dir, resolution, has_captions, all_captioned)

    shutil.rmtree(training_dir, ignore_errors=True)
    os.makedirs(training_dir, exist_ok=True)
    shutil.rmtree(mask_dir, ignore_errors=True)
    os.makedirs(mask_dir, exist_ok=True)
    shutil.rmtree(face_dir, ignore_errors=True)
    os.makedirs(face_dir, exist_ok=True)

    txt_hash = {}
    images_hash = {}
    for h in tune.file_names:
        url_filename = h.url.split('/')[-1]
        base_name, ext = os.path.splitext(h.filename)
        if ext == '.txt':
            txt_hash[url_filename] = base_name
            has_captions = True
        else:
            images_hash[base_name] = url_filename

    for i in range(10):
        batch_size = 50
        for i in range(0, len(tune.orig_images), batch_size):
            batch = tune.orig_images[i:i+batch_size]
            run(['curl', '-L', '--remote-name-all', '--parallel', '--retry', '25', '--retry-delay', '5', '--retry-all-errors', '--fail',  *batch], cwd=training_dir)
        if len(os.listdir(training_dir)) == len(tune.orig_images):
            break
        print(f"Failed to download all images for {tune.id}, retrying in 5 seconds")
        time.sleep(5)
    if len(os.listdir(training_dir)) != len(tune.orig_images):
        raise Exception(f"Failed to download all images for {tune.id}")

    birefnet = BiRefNet_node()
    face_crop = tune.face_crop and not tune.disable_face_crop
    if face_crop:
        yolo = YOLO(YOLO_FACE_MODEL)

    skipped_images = []
    blur_factors_map = {}
    failed_image_count = 0
    for orig_fn in tqdm(tune.orig_images, desc="Preprocessing images (segmentation)"):
        fn = orig_fn.split('/')[-1]
        if fn in txt_hash:
            orig_fn = txt_hash.pop(fn, None)
            if orig_fn:
                image_filename = images_hash[orig_fn]
                print(f"Renaming {fn} to {image_filename}.txt")
                shutil.copy(f"{training_dir}/{fn}", f"{training_dir}/{image_filename}-padded.txt")
                os.rename(f"{training_dir}/{fn}", f"{face_dir}/{image_filename}-center-crop.txt")
            else:
                print(f"Skipping {fn} as it does not have a corresponding image")
            continue

        try:
            image = io2img(f"{training_dir}/{fn}")
            image = enhance_luma_only(image)
        except Exception as e:
            print(f"Failed to convert {fn} to PNG: {e}")
            failed_image_count += 1
            continue
        finally:
            os.unlink(f"{training_dir}/{fn}")

        mask = get_mask(image, birefnet, mask_dir, fn)

        # Crop only to fit mask foreground
        image, mask = crop_mask_to_square(image, mask)

        if tune.black_mask:
            image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)

        augmenter = build_augmenter(resolution)
        save_pair = partial(
            save_img_augs,
            augmenter=augmenter,
            mask_dir=mask_dir)

        # 1. Center crop to resolution or Face crop
        bbox = None
        if face_crop:
            bbox, orig_bbox = get_face_bbox(image, yolo)

            # ---------- original bookkeeping (kept intact) --------------------------
            if is_slr(image):
                blur_factors_map[fn] = 200
                print(f"Detected SLR image {fn}")
            elif bbox and tune.name in HUMAN_CLASS_NAMES:
                bbox_size  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                image_size = image.width * image.height
                bbox_ratio = bbox_size / image_size

                # Evaluate sharpness on a standard 512 render of the face region
                cropped_face = fit_img(image.crop(orig_bbox), (512, 512))
                print(f"bbox_ratio={bbox_ratio} fn={fn}")
                blur_factor   = is_blurry(cropped_face)
                joint_factor  = (bbox_ratio + blur_factor) / 2
                print(
                    f"{fn=} bbox_ratio={bbox_ratio:.2f} "
                    f"blur_factor={blur_factor:.2f} joint_factor={joint_factor:.2f}"
                )
                blur_factors_map[fn] = blur_factor
            else:
                blur_factors_map[fn] = 200
            # -----------------------------------------------------------------------

            if bbox:  # ---- face found ----
                # CHANGED: do NOT pre-resize before augmenter
                img_crop = image.crop(bbox)

                if tune.only_face and tune.name in HUMAN_CLASS_NAMES:
                    center_crop_mask = Image.new('L', mask.size)
                    center_crop_mask.paste(
                        mask.crop(orig_bbox),
                        (int(orig_bbox[0]), int(orig_bbox[1]))
                    )
                else:
                    center_crop_mask = mask

                # CHANGED: do NOT pre-resize mask before augmenter
                mask_crop = center_crop_mask.crop(bbox)

                save_pair(
                    img_crop,
                    f"{face_dir}/{fn}-center-crop.png",
                    mask=mask_crop
                )

            elif tune.name not in HUMAN_CLASS_NAMES:  # ---- non-human class ----
                # CHANGED: pass native image & mask
                save_pair(
                    image,
                    f"{face_dir}/{fn}-center-crop.png",
                    mask=mask
                )

            else:  # ---- human but no face ----
                skipped_images.append(orig_fn)
                del blur_factors_map[fn]
                print(f"Skipping {fn} as no face was found")

        else:  # ---- no face-crop mode ----
            # CHANGED: pass native image & mask
            save_pair(
                image,
                f"{face_dir}/{fn}-center-crop.png",
                mask=mask
            )

        # 2. Optional padded copy for masked-loss training ---------------------------
        if (not face_crop) or bbox or (tune.name not in HUMAN_CLASS_NAMES):
            if tune.segmentation:
                # keep this as a separate padded variant (one intentional resample)
                img_pad = pad_img(image, (resolution, resolution))

                if (
                    face_crop and bbox and tune.only_face
                    and tune.name in HUMAN_CLASS_NAMES
                ):
                    new_mask = Image.new('L', mask.size)
                    new_mask.paste(
                        mask.crop(orig_bbox),
                        (int(orig_bbox[0]), int(orig_bbox[1]))
                    )
                    mask_for_pad = new_mask
                else:
                    mask_for_pad = mask

                mask_pad = pad_mask(mask_for_pad, (resolution, resolution))

                save_pair(
                    img_pad,
                    f"{training_dir}/{fn}-padded.png",
                    mask=mask_pad
                )
            else:
                # Prefer pad (letterbox) for the padded copy
                save_pair(
                    pad_img(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png"
                )

    # remove skipped images from orig_images so that train_batch calculation in train.py is correct
    if len(skipped_images):
        tune.orig_images = [fn for fn in tune.orig_images if fn not in skipped_images]
        print(f"Skipped {len(skipped_images)} images")

    if failed_image_count>2 or (failed_image_count>0 and len(tune.orig_images) < 4):
        print(f"Failed to process {failed_image_count} images")
        raise Exception(f"Failed to process {failed_image_count} images")
    else:
        print(f"Processed {len(tune.orig_images)} images. Skipped {failed_image_count} images")

    if face_crop:
        sorted_images = sorted(blur_factors_map.items(), key=lambda x: x[1], reverse=True)
        new_images = []
        for i, (fn, factor) in enumerate(sorted_images):
            threshold_factor = min(135,  120 + (len(new_images) - 4) * 5 // 8)
            if factor is not None and factor < threshold_factor and len(new_images) > 4:
                print(f"Removing {fn} with blur factor {factor:.0f} {threshold_factor=}")
                os.unlink(f"{face_dir}/{fn}-center-crop.png")
                os.unlink(f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png")
                os.unlink(f"{training_dir}/{fn}-padded.png")
            else:
                print(f"Keeping {fn} with blur factor {factor:.0f} {threshold_factor=}")
                new_images.append(fn)
        print(f"Keeping {len(new_images)} images after face crop. Before: {len(tune.orig_images)}")
        tune.orig_images = new_images


    if one_dir:
        return training_dir
    all_captioned = len(images_hash) == 0
    return DownloadTrainingOutput(training_dir, face_dir, mask_dir, resolution, has_captions, all_captioned)

def get_instance_prompt(tune, add_prefix=True):
    if os.environ.get('INSTANCE_PROMPT'):
        return os.environ.get('INSTANCE_PROMPT')
    elif tune.trigger:
        instance_prompt = tune.trigger
    elif tune.token and tune.name:
        instance_prompt = f"{tune.token} {tune.name}"
    elif tune.token and not tune.name:
        instance_prompt = tune.token
    elif not tune.token and tune.name:
        instance_prompt = tune.name
    else:
        raise Exception("Missing token or name")

    if 'style' not in tune.name and add_prefix:
        instance_prompt = f"A photo of {instance_prompt}"

    return instance_prompt

def write_metadata(training_dir , resolution, cache_file_suffix):
    files_training = sorted(list_full_paths(training_dir))
    with open(f'{training_dir}/aspect_ratio_bucket_metadata_{cache_file_suffix}.json', "w") as f:
        json.dump({
            fn: {
                "original_size": [resolution, resolution],
                "crop_coordinates": [0, 0],
                "target_size": [resolution, resolution],
                "intermediary_size": [resolution, resolution],
                "aspect_ratio": 1,
                "luminance": 100.0
            }
            for fn in files_training
        }, f)
    with open(f'{training_dir}/aspect_ratio_bucket_indices_{cache_file_suffix}.json', "w") as f:
        json.dump({
            "config": {
                "crop": True,
                "crop_aspect": "square",
                "crop_aspect_buckets": None,
                "crop_style": "center",
                "disable_validation": False,
                "resolution": resolution,
                "resolution_type": "pixel",
                "caption_strategy": "instanceprompt",
                "instance_data_dir": training_dir,
                "maximum_image_size": resolution,
                "target_downsample_size": resolution,
                "config_version": 2,
                "hash_filenames": True
            },
            "aspect_ratio_bucket_indices": {
                "1.0": files_training
            }
        }, f)

def create_data_config_wan(tune: JsonObj, output_dir: str, should_write_metadata=True) -> (str, int):
    ret = download_training(tune)
    resolution = ret.resolution
    tune.resolution = resolution

    instance_prompt = get_instance_prompt(tune)

    if tune.caption_strategy:
        caption_strategy = tune.caption_strategy
    elif ret.has_caption:
        caption_strategy = "textfile"
    else:
        caption_strategy = "instanceprompt"
        tune.caption_strategy = caption_strategy

    data = [
        {
            "id": "dreambooth-data",
            "type": "local",
            "dataset_type": "image",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution": resolution,
            "minimum_image_size": 64,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "resolution_type": "pixel",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux",
            "instance_data_dir": ret.training_dir,
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": caption_strategy,
            "instance_prompt": instance_prompt,
            "only_instance_prompt": caption_strategy == 'instanceprompt',
            "metadata_backend": "json",
            "cache_file_suffix": "square",
        },
        {
            "id": "dreambooth-data-face",
            "type": "local",
            "dataset_type": "image",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution": resolution,
            "minimum_image_size": 64,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "resolution_type": "pixel",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux-face",
            "instance_data_dir": ret.face_dir,
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": caption_strategy,
            "instance_prompt": "closeup " + instance_prompt,
            "only_instance_prompt": caption_strategy == 'instanceprompt',
            "metadata_backend": "json",
            "cache_file_suffix": "square",
        },
        {
            "id": "text-embeds",
            "type": "local",
            "dataset_type": "text_embeds",
            "default": True,
            "cache_dir": f"{output_dir}/cache-text",
            "disabled": False,
        }
    ]

    if should_write_metadata:
        write_metadata(ret.training_dir, resolution, 'square')
        write_metadata(ret.face_dir, resolution, 'square')

    if tune.segmentation:
        for i in range(2):
            data_copy = data[i]
            data_copy["id"] = data_copy["id"]+"-conditioned"
            data_copy["cache_dir_vae"] = data_copy["cache_dir_vae"] + "-conditioning"
            data_copy["conditioning_data"] = "dreambooth-conditioning"

        data.append({
            "id": "dreambooth-conditioning",
            "type": "local",
            "dataset_type": "conditioning",
            "instance_data_dir": ret.mask_dir,
            "resolution": resolution,
            "minimum_image_size": 64,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution_type": "pixel",
            "conditioning_type": "mask",
            "caption_strategy": caption_strategy,
            "instance_prompt": instance_prompt,
            "only_instance_prompt": caption_strategy == "instanceprompt",
            "cache_file_suffix": "square-mask",
        })

        if write_metadata:
            with open(f'{ret.mask_dir}/aspect_ratio_bucket_indices_square-mask.json', "w") as f:
                json.dump({
                    "config": {
                        "crop": True,
                        "crop_aspect": "square",
                        "crop_aspect_buckets": None,
                        "crop_style": "center",
                        "disable_validation": False,
                        "resolution": resolution,
                        "resolution_type": "pixel",
                        "caption_strategy": "instanceprompt",
                        "instance_data_dir": ret.mask_dir,
                        "maximum_image_size": resolution,
                        "target_downsample_size": resolution,
                        "config_version": 2,
                        "hash_filenames": True
                    },
                    "aspect_ratio_bucket_indices": {}
                }, f)

    if caption_strategy == "textfile" or ret.has_caption:
        data_textfile = data[0].copy()
        data_textfile["id"] = "dreambooth-textfile"
        data_textfile["caption_strategy"] = "textfile"
        data_textfile["only_instance_prompt"] = False
        data_textfile["instance_prompt"] = ""
        data_textfile["cache_dir_vae"] = f"{output_dir}/cache-vae-flux-textfile"
        data.append(data_textfile)

    if tune.multiresolution or os.environ.get('MULTIRESOLUTION'):
        all_resolutions = [768, 512]
        resolutions = [r for r in all_resolutions if r < resolution]
        for res_i, res in enumerate(resolutions):
            for i in range(2):
                data_copy = data[i].copy()
                data_copy["id"] = data_copy["id"] + f"-{res}"
                data_copy["cache_dir_vae"] = data_copy["cache_dir_vae"] + f"-{res}"
                data_copy["resolution"] = res
                data_copy["repeats"] = 5^(res_i+1)
                data_copy["target_downsample_size"] = res
                data_copy["cache_file_suffix"] = f"square-{res}"
                data.append(data_copy)
                write_metadata(ret.training_dir, res, f'square-{res}')
                write_metadata(ret.face_dir, res, f'square-{res}')

    if tune.regularization:
        pseudo_data_dir = "/data/cache/pseudo-camera-10k/canny"
        for d in data:
            d["repeats"] = ((d["repeats"] if 'repeats' in d else 0) + 1) * 1000
        regularization = {
            "id": "pseudo-camera-10k-flux",
            "type": "local",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution": 512,
            "minimum_image_size": 512,
            "maximum_image_size": 512,
            "target_downsample_size": 512,
            "resolution_type": "pixel_area",
            "cache_dir_vae": pseudo_data_dir + "-cache",
            "instance_data_dir": pseudo_data_dir,
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": "filename",
            "metadata_backend": "json",
            "repeats": 0,
            "is_regularisation_data": True,
        }
        data.append(regularization)

    print(json.dumps(data, indent=4))

    multidatabackend_config = f"{output_dir}/multidatabackend.json"
    with open(multidatabackend_config, "w") as f:
        json.dump(data, f, indent=4)

    if tune.caption_strategy == "textfile" and not os.environ.get('SKIP_DOWNLOAD'):
        run_with_output([
            'python3', 'astria/caption.py',
            ret.training_dir,
            '--autocaption-prefix',
            tune.autocaption_prefix or (f'Photo of {get_instance_prompt(tune, False)}' if 'style' not in tune.name else ''),
        ])
        for fn in os.listdir(ret.training_dir):
            if fn.endswith('.txt'):
                shutil.copy(f"{ret.training_dir}/{fn}", f"{ret.face_dir}/{fn}".replace('-padded', '-center-crop'))

    return multidatabackend_config, resolution


if __name__ == "__main__":
    import sys
    from train import parse_args
    if os.environ.get('MOCK_SERVER') or os.environ.get('DEBUG') == 'test':
        from astria_mock_server import request_tune_job_from_server
    else:
        from astria_server import request_tune_job_from_server
    for id in sys.argv[1:]:
        tune = request_tune_job_from_server(id)
        parse_args(tune)
        output_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-{tune.branch}"
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        create_data_config_v2(tune, output_dir)
        print(f"Downloaded training data for {tune.id}")
