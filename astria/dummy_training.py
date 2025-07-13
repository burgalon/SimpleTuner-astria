#!/usr/bin/env python3
import os
import json
from PIL import Image

def create_dummy_data_config(output_dir: str) -> tuple[str, int]:
    """
    Create a dummy dataset of 8 black images at 512×512 and
    write a minimal multi-data-backend config.

    Returns:
        (config_path, resolution)
    """
    # 1) Parameters
    num_images = 8
    resolution = 512
    os.makedirs(output_dir, exist_ok=True)

    # 2) Write 8 black images
    for idx in range(num_images):
        img = Image.new("RGB", (resolution, resolution), color=(0, 0, 0))
        img.save(os.path.join(output_dir, f"dummy_{idx}.png"))

    # 3) Write the "multidatabackend" JSON
    data = [
        {
            "id": "dummy-data",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": output_dir,
            "resolution": resolution,
            "minimum_image_size": resolution,
            "maximum_image_size": resolution,
            "target_downsample_size": resolution,
            "resolution_type": "pixel",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux",
            "disabled": False,
            "skip_file_discovery": "",
            "caption_strategy": "instanceprompt",
            "instance_prompt": "foo",
            "only_instance_prompt": True,
            "metadata_backend": "json",
            "cache_file_suffix": "dummy"
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
    config_path = os.path.join(output_dir, "multidatabackend.json")
    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)

    # print('-------------------------------------------------------------------')
    # print('WROTE TO ', config_path)
    # print('-------------------------------------------------------------------')

    return config_path, resolution


if __name__ == "__main__":
    # Example usage
    cfg_path, res = create_dummy_data_config("dummy_images")
    print(f"Written {res}×{res} black images to 'dummy_images/'")
    print(f"Data config at: {cfg_path}")
