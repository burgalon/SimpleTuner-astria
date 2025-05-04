import json
import os
import shutil
import time
import cv2
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


def list_full_paths(directory: str) -> list[str]:
    return [os.path.join(os.path.abspath(directory), f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]


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


CONST_BASE_TRAIN_RESOLUTION = 512
# This unforunately gets overriden inside the function in order to be used by get_face_bbox without passing argument
BASE_TRAIN_RESOLUTION = 512

# define Bbox
Bbox = tuple[float, float, float, float]

def get_face_bbox(image: Image.Image, yolo) -> (Bbox, Bbox):
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
    pred.orig_bboxes = [[x1-(x2-x1)*0.2, y1-(y2-y1)*.2, x2+(x2-x1)*0.2, y2+(y2-y1)*.1] for x1, y1, x2, y2 in pred.bboxes]
    pred.bboxes = [[x1-(x2-x1)*0.4, y1-(y2-y1)*.5, x2+(x2-x1)*0.4, y2+(y2-y1)*.5] for x1, y1, x2, y2 in pred.bboxes]
    if len(pred.bboxes) == 0:
        return (None, None)

    bbox = pred.bboxes[0]
    orig_bbox = pred.orig_bboxes[0]
    # Extend bbox to min BASE_TRAIN_RESOLUTION
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
    # move bbox y1 up by 10% to include hair and avoid overfitting to where the head is touching the top
    bbox = list(bbox)
    bbox[1] = max(0, bbox[1] - (bbox[3] - bbox[1])*.1)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # extend bbox so that the mask is padded a bit
    print(f"bbox before extend={bbox}")
    bbox = (max(0, bbox[0] - width//10), max(0, bbox[1] - height//10), min(image.width, bbox[2] + width//10), min(image.height, bbox[3] + height//10))
    print(f"bbox after extend={bbox}")
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
    dirs: list[str]
    mask_dir: str
    resolution: int
    has_caption: bool
    all_captioned: bool


def is_blurry(pil_image: Image.Image) -> float:
    """
    Computes a sharpness measure based on the Fourier transform.

    :param pil_image: A PIL Image object.
    :return: The sharpness measure (float). Higher values indicate a sharper image.
    """
    # Convert PIL to NumPy array
    open_cv_image = np.array(pil_image)
    # Convert from RGB to BGR for OpenCV
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply Fourier transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Return the mean of the magnitude spectrum
    return magnitude_spectrum.mean()

def is_slr(image: Image.Image) -> bool:
    # use exif to determine if image is portrait or landscape
    try:
        for key, value in image.getexif().items():
            if isinstance(value, str) and ('Canon' in value or 'Nikon' in value):
                return True
        return False
    except SyntaxError:
        # Avoid TIFF SyntaxError
        return False

def download_training(tune: JsonObj):
    global BASE_TRAIN_RESOLUTION
    resolution = int(tune.resolution) if tune.resolution else CONST_BASE_TRAIN_RESOLUTION
    has_captions = False
    all_captioned = False
    BASE_TRAIN_RESOLUTION = resolution
    training_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
    mask_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-masks"
    face_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-faces"
    body_mask_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-masks-body"

    if os.environ.get('SKIP_DOWNLOAD'):
        print("Skipping download as SKIP_DOWNLOAD is set")
        return DownloadTrainingOutput([training_dir, face_dir], mask_dir, resolution, has_captions, all_captioned)

    for dir in [training_dir, mask_dir, face_dir, body_mask_dir]:
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)

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
    # checking for HUMAN_CLASS_NAMES can be important to avoid cropping one face for class_name=couple
    face_crop = tune.face_crop and not tune.disable_face_crop and tune.name != 'couple'
    if face_crop:
        yolo = YOLO(YOLO_FACE_MODEL)

    skipped_images = []
    blur_factors_map = {}
    for orig_fn in tqdm(tune.orig_images, desc="Preprocessing images (segmentation)"):
        fn = orig_fn.split('/')[-1]
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
        os.unlink(f"{training_dir}/{fn}")

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
            bbox, orig_bbox = get_face_bbox(image, yolo)
            if is_slr(image):
                blur_factors_map[fn] = 200
                print(f"Detected SLR image {fn}")
            elif bbox and tune.name in HUMAN_CLASS_NAMES:
                bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                image_size = image.width * image.height
                bbox_ratio = bbox_size / image_size
                cropped_face = ImageOps.fit(image.crop(orig_bbox), (512, 512))
                print(f"bbox_ratio={bbox_ratio} fn={fn}")
                blur_factor = is_blurry(cropped_face)
                joint_factor = (bbox_ratio + blur_factor) / 2
                print(f"{fn=} bbox_ratio={bbox_ratio:.2f} blur_factor={blur_factor:.2f} joint_factor={joint_factor:.2f}")
                blur_factors_map[fn] = blur_factor
            else:
                blur_factors_map[fn] = 200 # high value to ensure that None values are at the end

            if bbox:
                save_img(
                    ImageOps.fit(image.crop(bbox), (resolution, resolution)),
                    f"{face_dir}/{fn}-center-crop.png"
                )

                # checking for HUMAN_CLASS_NAMES can be important to avoid cropping one face for class_name=couple
                if tune.only_face and tune.name in HUMAN_CLASS_NAMES:
                    # Train only on the face
                    # We want to keep the size of the original face image so that the model doesn't
                    # generate full closeups all the time
                    center_crop_mask = Image.new('L', mask.size)
                    center_crop_mask.paste(mask.crop(orig_bbox), (int(orig_bbox[0]), int(orig_bbox[1])))
                else:
                    center_crop_mask = mask
                save_img(
                    ImageOps.fit(center_crop_mask.crop(bbox) if face_crop else mask, (resolution, resolution)),
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
                skipped_images.append(orig_fn)
                del blur_factors_map[fn]
                print(f"Skipping {fn} as no face was found")
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
                # checking for HUMAN_CLASS_NAMES can be important to avoid cropping one face for class_name=couple
                if face_crop and bbox and tune.only_face:
                    if tune.augment_body:
                        save_img(
                            ImageOps.fit(mask, (resolution, resolution)),
                            f"{body_mask_dir}/{os.path.splitext(fn)[0]}-padded-body.png"
                        )
                    # Train only on the face
                    new_mask = Image.new('L', mask.size)
                    new_mask.paste(mask.crop(orig_bbox), (int(orig_bbox[0]), int(orig_bbox[1])))
                    mask = new_mask
                save_img(
                    ImageOps.pad(mask, (resolution, resolution)),
                    f"{mask_dir}/{os.path.splitext(fn)[0]}-padded.png"
                )
            else:
                save_img(
                    ImageOps.fit(image, (resolution, resolution)),
                    f"{training_dir}/{fn}-padded.png"
                )


    # remove skipped images from orig_images so that train_batch calculation in train.py is correct
    if len(skipped_images):
        tune.orig_images = [fn for fn in tune.orig_images if fn not in skipped_images]
        print(f"Skipped {len(skipped_images)} images")

    if face_crop: #  and tune.name in HUMAN_CLASS_NAMES:
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


    all_captioned = len(images_hash) == 0
    return DownloadTrainingOutput([training_dir, face_dir], mask_dir, resolution, has_captions, all_captioned)

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

def create_data_config_v3(tune: JsonObj, output_dir: str) -> (str, int):
    ret = download_training(tune)
    resolution = ret.resolution
    tune.resolution = resolution
    face_crop = tune.face_crop and not tune.disable_face_crop and tune.name != 'couple'

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
            "id": "text-embeds",
            "type": "local",
            "dataset_type": "text_embeds",
            "default": True,
            "cache_dir": f"{output_dir}/cache-text",
            "disabled": False,
            # "write_batch_size": 128
        }
    ]
    for dir in ret.dirs:
        dir_data = {
            "id": f"dreambooth-data-{os.path.basename(dir)}",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": dir,
            "resolution": resolution,
            "minimum_image_size": 64,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution_type": "pixel",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux-{os.path.basename(dir)}",
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": caption_strategy,
            "instance_prompt": "closeup " + instance_prompt if 'face' in dir else instance_prompt,
            "only_instance_prompt": caption_strategy == "instanceprompt",
            "metadata_backend": "json",
            "cache_file_suffix": "square",
            "conditioning_data": "dreambooth-conditioning" if tune.segmentation else None,
        }
        data.append(dir_data)
        write_metadata(dir, resolution, 'square')


    if tune.segmentation:
        # https://github.com/bghira/SimpleTuner/blob/main/documentation/DREAMBOOTH.md#masked-loss
        # duplicate first two data entries and add conditioning_data
        data_mask = {
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
        }
        data.append(data_mask)
        if tune.augment_body and face_crop:
            for d in data:
                if 'resolution' in d:
                    d['repeats'] = int(tune.augment_body)

            training_dir = ret.dirs[0]
            assert 'training' in training_dir

            # copy training_dir - add 'body' of the dir and name of each file
            body_training_dir = training_dir + "-body"
            shutil.rmtree(body_training_dir, ignore_errors=True)
            os.makedirs(body_training_dir, exist_ok=True)
            for fn in os.listdir(training_dir):
                if not fn.endswith('.png'):
                    continue
                shutil.copy(f"{training_dir}/{fn}", f"{body_training_dir}/{os.path.splitext(fn)[0]}-body.png")
                print(f"Copying {fn} to {body_training_dir}/{os.path.splitext(fn)[0]}-body.png")


            # Add mask data to JSON
            body_mask_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-masks-body"
            body_mask_data = data_mask.copy()
            body_mask_data["id"] += "-body"
            body_mask_data["instance_data_dir"] = body_mask_dir
            body_mask_data["repeats"] = 1
            body_mask_data["cache_file_suffix"] = "square-body-mask"
            # Add training data to JSON but conditioned on the body mask and lower repeats
            training_data = next(d for d in data if 'instance_data_dir' in d and d['instance_data_dir']==training_dir)
            assert 'training' in training_data['instance_data_dir']
            body_training_data = training_data.copy()
            body_training_data["id"] += "-body-mask"
            body_training_data["conditioning_data"] = body_mask_data["id"]
            body_training_data["instance_data_dir"] = body_training_dir
            body_training_data["repeats"] = 1
            body_training_data["cache_file_suffix"] = "square-body"
            body_training_data["cache_dir_vae"] = body_training_data["cache_dir_vae"] + "-body"
            data.append(body_mask_data)
            data.append(body_training_data)
            write_metadata(body_training_dir, resolution, 'square-body')




        write_metadata(ret.mask_dir, resolution, 'square-mask')


    if tune.multiresolution or os.environ.get('MULTIRESOLUTION'):
        # ordered is reversed so that repeats is higher for lower resolutions
        all_resolutions = [768, 512]
        resolutions = [r for r in all_resolutions if r < resolution]
        to_append = []
        for res_i, res in enumerate(resolutions):
            for data_i in data:
                if data_i['dataset_type']  != 'image':
                    continue
                print(f"Adding resolution {res} for {data_i['id']}")
                data_copy = data_i.copy()
                data_copy["id"] = data_i["id"] + f"-{res}"
                data_copy["cache_dir_vae"] = data_i["cache_dir_vae"] + f"-{res}"
                data_copy["resolution"] = res
                data_copy["repeats"] = 5^(res_i+1)
                # data_copy["minimum_image_size"] = 64
                # data_copy["maximum_image_size"] = res
                data_copy["target_downsample_size"] = res
                data_copy["cache_file_suffix"] = f"square-{res}"
                write_metadata(data_copy['instance_data_dir'], res, f'square-{res}')
                to_append.append(data_copy)
        data.extend(to_append)

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
    from train import parse_args, parse_env_args

    if os.environ.get('MOCK_SERVER') or os.environ.get('DEBUG') == 'test':
        from astria_mock_server import request_tune_job_from_server
    else:
        from astria_server import request_tune_job_from_server
    for id in sys.argv[1:]:
        tune = request_tune_job_from_server(id)
        parse_args(tune)
        parse_env_args(tune)
        # print('DEBUGGING WITH ONE IMAGE!!')
        # tune.orig_images = tune.orig_images[:1] # for testing
        output_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-{tune.branch}"
        print(f"augment_body={tune.augment_body}")
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        create_data_config_v3(tune, output_dir)
        print(f"Downloaded training data for {tune.id}")
