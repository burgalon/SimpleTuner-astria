import json
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO

from astria_utils import run, run_with_output, EPHEMERAL_MODELS_DIR, JsonObj, device, HUMAN_CLASS_NAMES
from birefnet.BiRefNet_node import BiRefNet_node
from image_utils import io2img, save_img
from inpaint_face_mixin import ultralytics_predict, filter_by_ratio, filter_k_largest, YOLO_FACE_MODEL


def infer_background(image: Image.Image, birefnet: BiRefNet_node) -> np.ndarray:
    alpha_tensor : torch.Tensor = birefnet.matting(image, 'cuda')
    alpha = alpha_tensor.squeeze().cpu().numpy()
    return alpha

def get_mask(image: Image.Image, birefnet: BiRefNet_node, mask_dir: str, image_name: str) -> None:
    """
    Generates a mask for the given image using BiRefNet and saves it to the specified directory.

    :param image: The input image for which to generate the mask.
    :param birefnet: An instance of the BiRefNet_node for segmentation.
    :param mask_dir: Directory to save the generated mask image.
    :param image_name: The name of the image file.
    """
    alpha = infer_background(image, birefnet)
    mask = (alpha >= 0.5).astype(np.uint8) * 255  # Create binary mask
    mask_image = Image.fromarray(mask)
    return mask_image
    # mask_path = os.path.join(mask_dir, f"{os.path.splitext(image_name)[0]}.png")
    # mask_image.save(mask_path)

BASE_TRAIN_RESOLUTION = 512

def get_face_bbox(image: Image.Image, yolo) -> Image.Image:
    pred = ultralytics_predict(
        yolo,
        image=image,
        confidence=0.3,
        device=device,
        classes=None,  # ad_model_classes,
    )
    pred = filter_by_ratio(pred, low=0.003, high=1)
    pred = filter_k_largest(pred, k=0)
    # increase bbox y2 by 40-50% to include full face with hair, chin, neck, ears
    # must be symmetrical because we're center cropping
    pred.bboxes = [[x1-(x2-x1)*0.4, y1-(y2-y1)*.5, x2+(x2-x1)*0.4, y2+(y2-y1)*.5] for x1, y1, x2, y2 in pred.bboxes]
    if len(pred.bboxes) == 0:
        return None

    bbox = pred.bboxes[0]
    # Extend bbox to min BASE_TRAIN_RESOLUTION
    if bbox[2] - bbox[0] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[2] - bbox[0])
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])
    if bbox[3] - bbox[1] < BASE_TRAIN_RESOLUTION:
        diff = BASE_TRAIN_RESOLUTION - (bbox[3] - bbox[1])
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)

    return bbox if bbox else None


def crop_mask_to_square(image: Image.Image, mask: Image.Image) -> [Image.Image, Image.Image]:
    bbox = mask.getbbox()
    if not bbox:
        print("crop_mask_to_square(): Failed to get bbox for mask")
        return image, mask
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # extend bbox while trying to get to square aspect ratio
    if width > height:
        diff = width - height
        bbox = (bbox[0], bbox[1] - diff//2, bbox[2], bbox[3] + diff//2)
    elif height > width:
        diff = height - width
        bbox = (bbox[0] - diff//2, bbox[1], bbox[2] + diff//2, bbox[3])
    # extend to min BASE_TRAIN_RESOLUTION
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
    birefnet = BiRefNet_node()
    face_crop = tune.face_crop and not tune.disable_face_crop
    if face_crop:
        yolo = YOLO(YOLO_FACE_MODEL)

    for fn in tqdm(tune.orig_images, desc="Preprocessing images (segmentation)"):
        fn = fn.split('/')[-1]
        if fn in txt_hash:
            # rename it to the respective image name but with txt extension
            orig_fn = txt_hash.pop(fn, None)
            if orig_fn:
                image_filename = images_hash[orig_fn]
                print(f"Renaming {fn} to {image_filename}.txt")
                # copy for -padded
                shutil.copy(f"{training_dir}/{fn}", f"{training_dir}/{image_filename}-padded.txt")
                os.rename(f"{training_dir}/{fn}", f"{face_dir}/{image_filename}-center-crop.txt")
            else:
                print(f"Skipping {fn} as it does not have a corresponding image")
            continue

        # convert to PNG and apply exif
        image = io2img(f"{training_dir}/{fn}")
        # if image.height < resolution/2 or image.width < resolution/2:
        #     print(f"Setting resolution to 512 for {fn} size={image.height}x{image.width}")
        #     resolution = min(512, resolution)

        mask = get_mask(image, birefnet, mask_dir, fn)

        # Crop only to fit mask foreground
        image, mask = crop_mask_to_square(image, mask)
        # turn all pixels masked black to black
        if tune.black_mask:
            image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)

        # 1. Center crop to resolution or Face crop
        if face_crop:
            bbox = get_face_bbox(image, yolo)
            if bbox:
                save_img(
                    ImageOps.fit(image.crop(bbox), (resolution, resolution)),
                    f"{face_dir}/{fn}-center-crop.png"
                )
                save_img(
                    ImageOps.fit(mask.crop(bbox) if face_crop else mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png"
                )
            elif tune.name not in HUMAN_CLASS_NAMES:
                print(f"Failed to find face for {fn}")
                save_img(
                    ImageOps.fit(image, (resolution, resolution)),
                    f"{face_dir}/{fn}-center-crop.png"
                )
                save_img(
                    ImageOps.fit(mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png"
                )
        else:
            save_img(
                ImageOps.fit(image, (resolution, resolution)),
                f"{face_dir}/{fn}-center-crop.png"
            )
            save_img(
                ImageOps.fit(mask, (resolution, resolution)),
                f"{mask_dir}/{os.path.splitext(fn)[0]}-center-crop.png"
            )



        # If face detection is enabled but no face was found, skip this image
        # This helps with cases of users uploading images by mistake of screenshots or pets that
        # should not be part of the training set and are a plain mistake
        if (not face_crop) or bbox or (tune.name not in HUMAN_CLASS_NAMES):
            # providing padded images without segmentation black border frames to show
            if tune.segmentation:
                # Pad to resolution
                save_img(
                    ImageOps.pad(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png"
                )
                save_img(
                    ImageOps.pad(mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-padded.png"
                )
            else:
                save_img(
                    ImageOps.fit(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png"
                )

        os.unlink(f"{training_dir}/{fn}")

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


def create_data_config_v2(tune: JsonObj, output_dir: str) -> (str, int):
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
            "write_batch_size": 128
        }
    ]

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
        for res in [512, 768]:
            for i in range(2):
                data_copy = data[i].copy()
                data_copy["id"] = data_copy["id"] + f"-{res}"
                data_copy["cache_dir_vae"] = data_copy["cache_dir_vae"] + f"-{res}"
                data_copy["resolution"] = res
                # data_copy["minimum_image_size"] = 64
                # data_copy["maximum_image_size"] = res
                data_copy["target_downsample_size"] = res
                data_copy["cache_file_suffix"] = f"square-{res}"
                data.append(data_copy)


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
    if os.environ.get('MOCK_SERVER'):
        from astria_mock_server import request_tune_job_from_server
    else:
        from astria_server import request_tune_job_from_server
    for id in sys.argv[1:]:
        tune = request_tune_job_from_server(id)
        output_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-{tune.branch}"
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        create_data_config_v2(tune, output_dir)
        print(f"Downloaded training data for {tune.id}")
