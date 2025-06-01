#!/usr/bin/env python3
"""
Multiprocess refactor of the Astria preprocessing script.

*   Each worker process loads its own BiRefNet and (optionally) YOLO face
    detector **once** in a per-process initializer, so the GPU models are
    not re-loaded for every image.
*   The number of subprocesses is controlled by the environment variable
    PARALLEL_NUMBER_SUBPROCESSES (default 4).  On roomy GPUs you can set it
    as high as 12.
*   The rest of the public API and CLI remains unchanged.
"""
# ----------------------------------------------------------------------
# ─── IMPORTS ───────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
import json
import os
import shutil
import time
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO

from astria_utils import (
    run,
    run_with_output,
    EPHEMERAL_MODELS_DIR,
    JsonObj,
    device,
    HUMAN_CLASS_NAMES,
)
from birefnet.BiRefNet_node import BiRefNet_node
from image_utils import io2img, save_img
from inpaint_face_mixin import (
    ultralytics_predict,
    filter_by_ratio,
    filter_k_largest,
    YOLO_FACE_MODEL,
)

mp.set_start_method("spawn", force=True)

# ----------------------------------------------------------------------
# ─── CONSTANTS & HELPERS ───────────────────────────────────────────────
# ----------------------------------------------------------------------
BASE_TRAIN_RESOLUTION = 512
PARALLEL_NUMBER_SUBPROCESSES = int(os.getenv("PARALLEL_NUMBER_SUBPROCESSES", 4))

# Type hint for a bounding box
Bbox = tuple[float, float, float, float]

def list_full_paths(directory: str) -> list[str]:
    return [
        os.path.join(os.path.abspath(directory), f)
        for f in os.listdir(directory)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

def infer_background(image: Image.Image, birefnet: BiRefNet_node) -> np.ndarray:
    alpha_tensor: torch.Tensor = birefnet.matting(image, "cuda")
    alpha = alpha_tensor.squeeze().cpu().numpy()
    return alpha

def get_mask(
    image: Image.Image, birefnet: BiRefNet_node, mask_dir: str, image_name: str
) -> Image.Image:
    alpha = infer_background(image, birefnet)
    mask = (alpha >= 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)

def get_face_bbox(image: Image.Image, yolo) -> tuple[Bbox | None, Bbox | None]:
    pred = ultralytics_predict(
        yolo, image=image, confidence=0.3, device=device, classes=None
    )
    pred = filter_by_ratio(pred, low=0.003, high=1)
    pred = filter_k_largest(pred, k=0)
    pred.orig_bboxes = [
        [
            x1 - (x2 - x1) * 0.2,
            y1 - (y2 - y1) * 0.2,
            x2 + (x2 - x1) * 0.2,
            y2 + (y2 - y1) * 0.1,
        ]
        for x1, y1, x2, y2 in pred.bboxes
    ]
    pred.bboxes = [
        [
            x1 - (x2 - x1) * 0.4,
            y1 - (y2 - y1) * 0.5,
            x2 + (x2 - x1) * 0.4,
            y2 + (y2 - y1) * 0.5,
        ]
        for x1, y1, x2, y2 in pred.bboxes
    ]
    if len(pred.bboxes) == 0:
        return None, None

    bbox = pred.bboxes[0]
    orig_bbox = pred.orig_bboxes[0]

    if bbox[2] - bbox[0] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[2] - bbox[0])
        bbox = (bbox[0] - diff // 2, bbox[1], bbox[2] + diff // 2, bbox[3])
    if bbox[3] - bbox[1] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[3] - bbox[1])
        bbox = (bbox[0], bbox[1] - diff // 2, bbox[2], bbox[3] + diff // 2)

    return bbox, orig_bbox

def crop_mask_to_square(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    bbox = mask.getbbox()
    if not bbox:
        print("crop_mask_to_square(): Failed to get bbox for mask")
        return image, mask
    bbox = list(bbox)
    bbox[1] = max(0, bbox[1] - (bbox[3] - bbox[1]) * 0.1)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width > height:
        diff = width - height
        bbox = (bbox[0], bbox[1] - diff // 2, bbox[2], bbox[3] + diff // 2)
    elif height > width:
        diff = height - width
        bbox = (bbox[0] - diff // 2, bbox[1], bbox[2] + diff // 2, bbox[3])
    if bbox[2] - bbox[0] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[2] - bbox[0])
        bbox = (bbox[0] - diff // 2, bbox[1], bbox[2] + diff // 2, bbox[3])
    if bbox[3] - bbox[1] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[3] - bbox[1])
        bbox = (bbox[0], bbox[1] - diff // 2, bbox[2], bbox[3] + diff // 2)
    return image.crop(bbox), mask.crop(bbox)

def is_blurry(pil_image: Image.Image) -> float:
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    return mag.mean()

def is_slr(image: Image.Image) -> bool:
    try:
        for _k, v in image.getexif().items():
            if isinstance(v, str) and ("Canon" in v or "Nikon" in v):
                return True
        return False
    except SyntaxError:
        return False

# ----------------------------------------------------------------------
# ─── PER-PROCESS INITIALISER ───────────────────────────────────────────
# ----------------------------------------------------------------------
_birefnet: BiRefNet_node | None = None
_yolo: YOLO | None = None
_face_crop: bool = False

def _init_worker(face_crop: bool):
    """Initialises heavy GPU models once per subprocess."""
    global _birefnet, _yolo, _face_crop
    _birefnet = BiRefNet_node()
    _face_crop = face_crop
    if face_crop:
        _yolo = YOLO(YOLO_FACE_MODEL)

# ----------------------------------------------------------------------
# ─── WORKER FUNCTION ──────────────────────────────────────────────────
# ----------------------------------------------------------------------
def _process_image(args) -> dict:
    try:
        (
            orig_fn,
            training_dir,
            mask_dir,
            face_dir,
            resolution,
            tune_flags,          # ← simple dict, not the whole tune object
            txt_hash,
            images_hash,
        ) = args

        # unpack the flags so old code stays readable
        black_mask   = tune_flags["black_mask"]
        only_face    = tune_flags["only_face"]
        class_name   = tune_flags["name"]
        segmentation = tune_flags["segmentation"]

        fn = os.path.basename(orig_fn)
        result = {"fn": fn, "blur_factor": None, "skipped": False, "failed": False}

        # --- Handle text side-car files cheaply ---------------------------
        if fn in txt_hash:
            txt_orig = txt_hash.pop(fn, None)
            if txt_orig:
                img_base = images_hash[txt_orig]
                shutil.copy(
                    f"{training_dir}/{fn}", f"{training_dir}/{img_base}-padded.txt"
                )
                os.rename(
                    f"{training_dir}/{fn}", f"{face_dir}/{img_base}-center-crop.txt"
                )
            return result  # done for .txt

        # --- Load & delete original file ----------------------------------
        try:
            image = io2img(f"{training_dir}/{fn}")
        except Exception:
            result["failed"] = True
            return result
        finally:
            try:
                os.unlink(f"{training_dir}/{fn}")
            except FileNotFoundError:
                pass

        # --- Segmentation & cropping --------------------------------------
        mask = get_mask(image, _birefnet, mask_dir, fn)
        image, mask = crop_mask_to_square(image, mask)
        if black_mask:
            image = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )

        bbox = orig_bbox = None
        if _face_crop:
            bbox, orig_bbox = get_face_bbox(image, _yolo)

        # --- Face / centre crop branch ------------------------------------
        if _face_crop:
            if is_slr(image):
                blur_factor = 200.0
            elif bbox and class_name in HUMAN_CLASS_NAMES:
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                img_area = image.width * image.height
                bbox_ratio = bbox_area / img_area
                cropped_face = ImageOps.fit(image.crop(orig_bbox), (512, 512))
                blur_factor = is_blurry(cropped_face)
                print(
                    f"{fn=} bbox_ratio={bbox_ratio:.2f} "
                    f"blur_factor={blur_factor:.2f}"
                )
            else:
                blur_factor = 200.0

            result["blur_factor"] = blur_factor

            if bbox:
                save_img(
                    ImageOps.fit(image.crop(bbox), (resolution, resolution)),
                    f"{face_dir}/{fn}-center-crop.png",
                )

                if only_face and class_name in HUMAN_CLASS_NAMES:
                    center_crop_mask = Image.new("L", mask.size)
                    center_crop_mask.paste(
                        mask.crop(orig_bbox), (int(orig_bbox[0]), int(orig_bbox[1]))
                    )
                else:
                    center_crop_mask = mask

                save_img(
                    ImageOps.fit(center_crop_mask.crop(bbox), (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png",
                )
            elif class_name not in HUMAN_CLASS_NAMES:
                save_img(
                    ImageOps.fit(image, (resolution, resolution)),
                    f"{face_dir}/{fn}-center-crop.png",
                )
                save_img(
                    ImageOps.fit(mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png",
                )
            else:
                result["skipped"] = True
                return result
        else:
            # face-crop disabled
            save_img(
                ImageOps.fit(image, (resolution, resolution)),
                f"{face_dir}/{fn}-center-crop.png",
            )
            save_img(
                ImageOps.fit(mask, (resolution, resolution)),
                f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png",
            )

        # --- Padded / segmentation copies ---------------------------------
        if (not _face_crop) or bbox or (class_name not in HUMAN_CLASS_NAMES):
            if segmentation:
                save_img(
                    ImageOps.pad(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png",
                )
                if _face_crop and bbox and only_face and class_name in HUMAN_CLASS_NAMES:
                    new_mask = Image.new("L", mask.size)
                    new_mask.paste(
                        mask.crop(orig_bbox), (int(orig_bbox[0]), int(orig_bbox[1]))
                    )
                    mask = new_mask
                save_img(
                    ImageOps.pad(mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-padded.png",
                )
            else:
                save_img(
                    ImageOps.fit(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png",
                )
    except Exception as e:
        import traceback
        # Send the failure back instead of killing the worker
        return {
            "fn": os.path.basename(args[0]),
            "failed": True,
            "error": repr(e),
            "stacktrace": traceback.format_exc(),
            "skipped": False,
            "blur_factor": None,
        }

    return result

# ----------------------------------------------------------------------
# ─── MULTI-PROCESS DRIVER ─────────────────────────────────────────────
# ----------------------------------------------------------------------
def process_images_mp(
    tune: JsonObj,
    training_dir: str,
    mask_dir: str,
    face_dir: str,
    resolution: int,
    txt_hash: dict,
    images_hash: dict,
):
    face_crop = tune.face_crop and not tune.disable_face_crop

    # everything the worker needs from `tune`, distilled to primitives
    tune_flags = {
        "black_mask":   getattr(tune, "black_mask",   False),
        "only_face":    getattr(tune, "only_face",    False),
        "name":         getattr(tune, "name",         ""),
        "segmentation": getattr(tune, "segmentation", False),
    }

    # build the list of argument tuples – one per original file
    tasks = [
        (
            orig_fn,
            training_dir,
            mask_dir,
            face_dir,
            resolution,
            tune_flags,
            txt_hash,
            images_hash,
        )
        for orig_fn in tune.orig_images
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=PARALLEL_NUMBER_SUBPROCESSES,
        initializer=_init_worker,
        initargs=(face_crop,),
    ) as pool:
        blur_factors_map: dict[str, float] = {}
        skipped_images: list[str] = []
        failed_image_count = 0

        for r in tqdm(
            pool.imap_unordered(_process_image, tasks),
            total=len(tune.orig_images),
            desc="Preprocessing images (segmentation, mp)",
        ):
            print('Process finished')
            if r.get("failed"):
                failed_image_count += 1
                print(f"[worker-error] {r['fn']}: {r.get('error')}")
                print(f"[worker-stacktrace] {r['fn']}: {r.get('stacktrace')}")
            elif r["skipped"]:
                skipped_images.append(r["fn"])
            if r["blur_factor"] is not None:
                blur_factors_map[r["fn"]] = r["blur_factor"]

    return blur_factors_map, skipped_images, failed_image_count

# ----------------------------------------------------------------------
# ─── REMAINDER OF ORIGINAL SCRIPT ─────────────────────────────────────
# ----------------------------------------------------------------------
# All original helper classes (DownloadTrainingOutput, etc.) and
# functions (write_metadata, create_data_config_v2, …) stay **unchanged**
# except for ONE line inside download_training(): we replace the slow
# loop with the call to `process_images_mp`.
# ----------------------------------------------------------------------
@dataclass
class DownloadTrainingOutput:
    training_dir: str
    face_dir: str
    mask_dir: str
    resolution: int
    has_caption: bool
    all_captioned: bool

# (the rest of download_training is identical up to the point where the
# old for-loop started)

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

    ## Create a hash for mapping captions to images
    txt_hash = {} # text files
    images_hash = {} # images
    for h in tune.file_names:
        url_filename = h.url.split('/')[-1]
        base_name, ext = os.path.splitext(h.filename)
        if ext == '.txt':
            txt_hash[url_filename] = base_name
            has_captions = True
        else:
            images_hash[base_name] = url_filename

    ## Download images
    for i in range(10):
        batch_size = 50
        for i in range(0, len(tune.orig_images), batch_size):
            batch = tune.orig_images[i:i+batch_size]
            run(['curl', '-L', '--remote-name-all', '--parallel', '--retry', '25', '--retry-delay', '5', '--retry-all-errors', '--fail',  *batch], cwd=training_dir)
        # IMPORTANT CRITICAL! otherwise --fail will just result in 3 files out of 20 for example.
        if len(os.listdir(training_dir)) == len(tune.orig_images):
            break
        print(f"Failed to download all images for {tune.id}, retrying in 5 seconds")
        time.sleep(5)
    if len(os.listdir(training_dir)) != len(tune.orig_images):
        raise Exception(f"Failed to download all images for {tune.id}")

    ## Preprocess images
    # birefnet = BiRefNet_node()
    # face_crop = tune.face_crop and not tune.disable_face_crop
    # if face_crop:
    #     yolo = YOLO(YOLO_FACE_MODEL)

    skipped_images = []
    blur_factors_map = {}
    failed_image_count = 0
    blur_factors_map, skipped_images, failed_image_count = process_images_mp(
        tune,
        training_dir,
        mask_dir,
        face_dir,
        resolution,
        txt_hash,
        images_hash,
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

    if tune.face_crop and not tune.disable_face_crop: #  and tune.name in HUMAN_CLASS_NAMES:
        # sort images by blur factor and remove all with factor<130
        # ensure at least 4 images are left - keep at least the sharpest images
        # 200 is a high value to ensure that None values are at the end
        sorted_images = sorted(blur_factors_map.items(), key=lambda x: x[1], reverse=True)
        new_images = []
        for i, (fn, factor) in enumerate(sorted_images):
            # example
            # threshold_factor = 120 if len(new_images) = 4
            # threshold_factor = 125 if len(new_images) = 12
            # threshold_factor = 130 if len(new_images) = 20
            # threshold_factor = 135 if len(new_images) = 28
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

def create_data_config_v2(tune: JsonObj, output_dir: str, should_write_metadata=True) -> (str, int):
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
            # "write_batch_size": 128
        }
    ]

    if should_write_metadata:
        write_metadata(ret.training_dir, resolution, 'square')
        write_metadata(ret.face_dir, resolution, 'square')

    if tune.segmentation:

        # https://github.com/bghira/SimpleTuner/blob/main/documentation/DREAMBOOTH.md#masked-loss
        # duplicate first two data entries and add conditioning_data
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

    # When using textfile augment the dataset with another copy so that
    # 50% is only instanceprompt and 50% is instanceprompt + textfile
    if caption_strategy == "textfile" or ret.has_caption:
        data_textfile = data[0].copy()
        data_textfile["id"] = "dreambooth-textfile"
        data_textfile["caption_strategy"] = "textfile"
        data_textfile["only_instance_prompt"] = False
        data_textfile["instance_prompt"] = ""
        data_textfile["cache_dir_vae"] = f"{output_dir}/cache-vae-flux-textfile"
        data.append(data_textfile)

    if tune.multiresolution or os.environ.get('MULTIRESOLUTION'):
        # ordered is reversed so that repeats is higher for lower resolutions
        all_resolutions = [768, 512]
        resolutions = [r for r in all_resolutions if r < resolution]
        for res_i, res in enumerate(resolutions):
            for i in range(2):
                data_copy = data[i].copy()
                data_copy["id"] = data_copy["id"] + f"-{res}"
                data_copy["cache_dir_vae"] = data_copy["cache_dir_vae"] + f"-{res}"
                data_copy["resolution"] = res
                data_copy["repeats"] = 5^(res_i+1)
                # data_copy["minimum_image_size"] = 64
                # data_copy["maximum_image_size"] = res
                data_copy["target_downsample_size"] = res
                data_copy["cache_file_suffix"] = f"square-{res}"
                data.append(data_copy)
                write_metadata(ret.training_dir, res, f'square-{res}')
                write_metadata(ret.face_dir, res, f'square-{res}')

    # requires tune.lora_type=lycoris
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
            "is_regularisation_data": True
        }
        data.append(regularization)
        # files_training = sorted(list_full_paths(pseudo_data_dir))
        # with open(f'{pseudo_data_dir}/aspect_ratio_bucket_metadata_square.json', "w") as f:
        #     json.dump({
        #         fn: {
        #             "original_size": [resolution, resolution],
        #             "crop_coordinates": [0, 0],
        #             "target_size": [resolution, resolution],
        #             "intermediary_size": [resolution, resolution],
        #             "aspect_ratio": 1,
        #             "luminance": 100.0
        #         }
        #         for fn in files_training
        #     }, f)
        # with open(f'{pseudo_data_dir}/aspect_ratio_bucket_indices_square.json', "w") as f:
        #     json.dump({
        #         "config": {
        #             "crop": True,
        #             "crop_aspect": "square",
        #             "crop_aspect_buckets": None,
        #             "crop_style": "center",
        #             "disable_validation": False,
        #             "resolution": resolution,
        #             "resolution_type": "pixel",
        #             "caption_strategy": "filename",
        #             "instance_data_dir": pseudo_data_dir,
        #             "maximum_image_size": resolution,
        #             "target_downsample_size": resolution,
        #             "config_version": 2,
        #             "hash_filenames": True
        #         },
        #         "aspect_ratio_bucket_indices": {
        #             "1.0": files_training
        #         }
        #     }, f)

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
        # copy captions to face dir
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
