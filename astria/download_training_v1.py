import json
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from astria_utils import run, run_with_output, EPHEMERAL_MODELS_DIR, JsonObj
from birefnet.BiRefNet_node import BiRefNet_node
from image_utils import io2img, save_img


def infer_background(image: Image.Image, birefnet: BiRefNet_node) -> np.ndarray:
    alpha_tensor : torch.Tensor = birefnet.matting(image, 'cuda')
    alpha = alpha_tensor.squeeze().cpu().numpy()
    # resize alpha to image size
    return alpha

def blackend_background(image: Image.Image, birefnet) -> Image.Image:
    alpha = infer_background(image, birefnet)
    image = np.array(image)
    image[alpha < 0.5] = 0
    return Image.fromarray(image)


def download_training(tune: JsonObj):
    training_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"

    if os.environ.get('SKIP_DOWNLOAD'):
        print("Skipping download as SKIP_DOWNLOAD is set")
        return training_dir, int(tune.resolution) if tune.resolution else 512

    shutil.rmtree(training_dir, ignore_errors=True)
    os.makedirs(training_dir, exist_ok=True)

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

    birefnet = BiRefNet_node()

    resolution = int(tune.resolution) if tune.resolution else 512
    for fn in tqdm(tune.orig_images, desc="Preprocessing images (segmentation)"):
        fn = fn.split('/')[-1]
        # convert to PNG and apply exif
        image = io2img(f"{training_dir}/{fn}")
        if image.height < 1024 or image.width < 1024:
            print(f"Setting resolution to 512 for {fn} size={image.height}x{image.width}")
            resolution = 512

        # Preprocess the image using BiRefNet
        if os.environ.get('SEGMENTATION') or tune.segmentation:
            image = blackend_background(image, birefnet)

        # Save the processed image
        save_img(image, f"{training_dir}/{fn}.png")
        os.unlink(f"{training_dir}/{fn}")

    return training_dir, resolution



def get_instance_prompt(tune):
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
    return instance_prompt


def create_data_config_v1(tune: JsonObj, output_dir: str) -> (str, int):
    instance_data_dir, resolution = download_training(tune)
    tune.resolution = resolution
    caption_strategy = tune.caption_strategy or "instanceprompt"

    instance_prompt = get_instance_prompt(tune)

    data = [
        {
            "id": f"{tune.id}-training",
            "type": "local",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "resolution": resolution,
            "minimum_image_size": 64,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "resolution_type": "pixel",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux",
            "instance_data_dir": instance_data_dir,
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": caption_strategy,
            "instance_prompt": instance_prompt,
            "only_instance_prompt": True,
            "metadata_backend": "json"
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
    multidatabackend_config = f"{output_dir}/multidatabackend.json"
    with open(multidatabackend_config, "w") as f:
        json.dump(data, f, indent=4)

    if not os.environ.get('SKIP_DOWNLOAD') and caption_strategy == "textfile":
        run_with_output([
            'python3', 'astria/caption.py',
            instance_data_dir,
            '--autocaption-suffix', tune.autocaption_prefix or (f'Photo of {get_instance_prompt(tune)}' if 'style' not in tune.name else '')
        ])

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
        create_data_config_v1(tune, output_dir)
        print(f"Downloaded training data for {tune.id}")
