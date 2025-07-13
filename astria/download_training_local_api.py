# File: download_training_local_api.py

import argparse
import base64
import concurrent.futures
import json
import os
import shutil
import sys

import requests
from tqdm import tqdm

# NOTE: Make sure these local modules are in your Python path.
from astria_utils import run, EPHEMERAL_MODELS_DIR, JsonObj, HUMAN_CLASS_NAMES

from astria.preprocess_server import BASE_PREPROCESS_PORT

def _list_full_paths(directory: str) -> list[str]:
    """Lists full paths of image files in a directory."""
    return [os.path.join(os.path.abspath(directory), f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]

def _get_instance_prompt(tune):
    """Generates the instance prompt string from the tune object."""
    if getattr(tune, "trigger", None):
        instance_prompt = tune.trigger
    elif tune.token and tune.name:
        instance_prompt = f"{tune.token} {tune.name}"
    elif tune.token:
        instance_prompt = tune.token
    elif tune.name:
        instance_prompt = tune.name
    else:
        raise Exception("Tune object is missing 'trigger', 'token', or 'name' to generate a prompt.")
    return instance_prompt

def _write_aspect_ratio_metadata(training_dir: str, resolution: int, cache_file_suffix: str):
    """Generates the aspect ratio bucket metadata files required by the trainer."""
    files_training = sorted(_list_full_paths(training_dir))
    
    metadata = {
        fn: {
            "original_size": [resolution, resolution],
            "crop_coordinates": [0, 0],
            "target_size": [resolution, resolution],
            "intermediary_size": [resolution, resolution],
            "aspect_ratio": 1.0,
            "luminance": 100.0
        } for fn in files_training
    }
    
    indices = {
        "config": {
            "resolution": resolution,
            "instance_data_dir": training_dir
        },
        "aspect_ratio_bucket_indices": {
            "1.0": files_training
        }
    }

    metadata_path = os.path.join(training_dir, f'aspect_ratio_bucket_metadata_{cache_file_suffix}.json')
    indices_path = os.path.join(training_dir, f'aspect_ratio_bucket_indices_{cache_file_suffix}.json')

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    with open(indices_path, "w") as f:
        json.dump(indices, f, indent=4)

# ----------------------------------------------------------------------
# ─── API INTERACTION ──────────────────────────────────────────────────
# ----------------------------------------------------------------------

def _process_single_image_via_api(image_url: str, tune: JsonObj, version: str, port: int):
    """Calls a specific API server for a single image with robust type handling."""
    processor_api_url = f"http://localhost:{port}/process/"
    is_v1 = version == 'v1'

    # --- ROBUST FIX APPLIED HERE ---
    # Safely get boolean values, providing a default of False if the attribute
    # is missing or is None. This prevents sending `null` for a boolean field.
    face_crop_enabled = getattr(tune, 'face_crop', False) and not getattr(tune, 'disable_face_crop', False)
    
    config = {
        "image_url": image_url,
        "resolution": int(tune.resolution or 512),
        "segmentation": bool(getattr(tune, "segmentation", True)),
        "use_bisenet": bool(getattr(tune, "use_bisenet", True)),
        "crop_expansion_factor": getattr(tune, "crop_expansion_factor", 0.0) or 0.0,
        "face_crop": False if is_v1 else face_crop_enabled,
        "only_face": False if is_v1 else bool(getattr(tune, "only_face", False)),
        "black_mask": bool(getattr(tune, "segmentation", False)) if is_v1 else bool(getattr(tune, "black_mask", False)),
        "is_human_subject": tune.name in HUMAN_CLASS_NAMES,
    }
    
    try:
        response = requests.post(processor_api_url, json=config, timeout=180)
        response.raise_for_status() # Raises an exception for 4xx/5xx errors
        data = response.json()
        data["original_url"] = image_url
        return data
    except requests.exceptions.RequestException as e:
        # --- IMPROVED ERROR LOGGING ---
        # This will print the detailed validation error from the FastAPI server.
        error_detail = ""
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except json.JSONDecodeError:
                error_detail = e.response.text
        
        print(f"\nAPI call to port {port} failed for {image_url}: {e}\n>>> SERVER RESPONSE: {error_detail}", file=sys.stderr)
        return {"original_url": image_url, "skipped": True, "images": {}, "blur_factor": None}


# ----------------------------------------------------------------------
# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────
# ----------------------------------------------------------------------

def get_instance_prompt(tune):
    """Generates the instance prompt string (common across versions)."""
    if getattr(tune, "trigger", None): return tune.trigger
    if tune.token and tune.name: return f"{tune.token} {tune.name}"
    if tune.token: return tune.token
    if tune.name: return tune.name
    raise Exception("Missing token or name for prompt generation.")

def list_full_paths(directory: str) -> list[str]:
    """Lists full paths of image files in a directory."""
    return [os.path.join(os.path.abspath(directory), f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]

def write_aspect_ratio_metadata(training_dir: str, resolution: int, cache_file_suffix: str):
    """Generates the aspect ratio bucket metadata files required by the trainers."""
    files = sorted(list_full_paths(training_dir))
    metadata = {
        fn: {
            "original_size": [resolution, resolution], "crop_coordinates": [0, 0],
            "target_size": [resolution, resolution], "intermediary_size": [resolution, resolution],
            "aspect_ratio": 1.0, "luminance": 100.0
        } for fn in files
    }
    indices = {
        "config": { "resolution": resolution, "instance_data_dir": training_dir },
        "aspect_ratio_bucket_indices": { "1.0": files }
    }
    with open(f'{training_dir}/aspect_ratio_bucket_metadata_{cache_file_suffix}.json', "w") as f:
        json.dump(metadata, f)
    with open(f'{training_dir}/aspect_ratio_bucket_indices_{cache_file_suffix}.json', "w") as f:
        json.dump(indices, f, indent=4)

# ----------------------------------------------------------------------
# ─── V1 EMULATION ─────────────────────────────────────────────────────
# ----------------------------------------------------------------------

def run_v1_processing(tune: JsonObj, output_dir: str, server_ports: list[int]):
    """Emulates download_training_v1.py by distributing work across available servers."""
    print("Running in v1 emulation mode...")

    if tune.resolution is None:
        tune.resolution = 512

    training_dir = f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training"
    shutil.rmtree(training_dir, ignore_errors=True)
    os.makedirs(training_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(server_ports) * 4) as executor:
        futures = [executor.submit(_process_single_image_via_api, url, tune, 'v1', server_ports[i % len(server_ports)]) for i, url in enumerate(tune.orig_images)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="🖼️  Processing (v1 Style)"):
            result = future.result()
            if result.get("skipped") or not result.get("images"): continue
            b64_data = result["images"]["padded_image"]
            fn = result['original_url'].split('/')[-1]
            out_path = os.path.join(training_dir, f"{os.path.splitext(fn)[0]}.png")
            with open(out_path, "wb") as f: f.write(base64.b64decode(b64_data))
    
    print("\n⚙️  Generating multidatabackend.json (v1 Style)...")
    instance_prompt = get_instance_prompt(tune)
    data = [
        {
            "id": f"{tune.id}-training", "type": "local", "crop": True,
            "crop_aspect": "square", "crop_style": "center",
            "resolution": tune.resolution, "instance_data_dir": training_dir,
            "caption_strategy": "instanceprompt", "instance_prompt": instance_prompt,
            "only_instance_prompt": True, "metadata_backend": "json",
            "cache_dir_vae": f"{output_dir}/cache-vae-flux",
        },
        { "id": "text-embeds", "type": "local", "dataset_type": "text_embeds", "default": True, "cache_dir": f"{output_dir}/cache-text" }
    ]
    
    multidatabackend_config = f"{output_dir}/multidatabackend.json"
    with open(multidatabackend_config, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n✅ v1 processing complete. Metadata saved to {multidatabackend_config}")

    return multidatabackend_config

# ----------------------------------------------------------------------
# ─── V2/V3 EMULATION ──────────────────────────────────────────────────
# ----------------------------------------------------------------------

def run_v2_v3_processing(tune: JsonObj, output_dir: str, version: str, server_ports: list[int]):
    """
    Emulates the full workflows of download_training_v2.py and v3.py by 
    calling the API service, filtering results, and generating a detailed data config.
    """
    print(f"🚀 Running in v{version} emulation mode...")

    # Ensure resolution is set
    if not hasattr(tune, 'resolution') or tune.resolution is None:
        tune.resolution = 512

    # Define and create all necessary directories
    dirs = {
        "training_dir": f"{EPHEMERAL_MODELS_DIR}/{tune.id}-training",
        "face_dir": f"{EPHEMERAL_MODELS_DIR}/{tune.id}-faces",
        "mask_dir": f"{EPHEMERAL_MODELS_DIR}/{tune.id}-masks",
    }
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # Call API servers in parallel to process all images
    blur_factors, processed_urls = {}, set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(server_ports) * 4) as executor:
        futures = [executor.submit(_process_single_image_via_api, url, tune, version, server_ports[i % len(server_ports)]) for i, url in enumerate(tune.orig_images)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="🖼️  Processing Images"):
            result = future.result()
            if result.get("skipped") or not result.get("images"):
                continue
            
            processed_urls.add(result["original_url"])
            if result.get("blur_factor") is not None:
                blur_factors[result["original_url"]] = result["blur_factor"]
            
            fn = result['original_url'].split('/')[-1]
            # Save all image variants to their respective directories
            with open(os.path.join(dirs['face_dir'], f"{fn}-center-crop.png"), "wb") as f: f.write(base64.b64decode(result["images"]["face_crop"]))
            with open(os.path.join(dirs['training_dir'], f"{fn}-padded.png"), "wb") as f: f.write(base64.b64decode(result["images"]["padded_image"]))
            with open(os.path.join(dirs['mask_dir'], f"{fn}-center-crop.png"), "wb") as f: f.write(base64.b64decode(result["images"]["face_mask"]))
            with open(os.path.join(dirs['mask_dir'], f"{fn}-padded.png"), "wb") as f: f.write(base64.b64decode(result["images"]["padded_mask"]))

    # Perform blur filtering based on the logic from your original scripts
    if getattr(tune, 'face_crop', False) and tune.name in HUMAN_CLASS_NAMES:
        print("\n🧐 Filtering images based on blurriness...")
        sorted_images = sorted(blur_factors.items(), key=lambda item: item[1], reverse=True)
        kept_urls = set()
        
        for i, (url, factor) in enumerate(sorted_images):
            # Dynamic threshold from your script
            threshold = min(135.0, 120.0 + (len(kept_urls) - 4) * 0.625)
            if factor < threshold and len(kept_urls) >= 4:
                fn_base = url.split('/')[-1]
                print(f"  -> Filtering out {fn_base} (blur factor: {factor:.2f} < {threshold:.2f})")
                # Remove all files associated with the filtered image
                for d in dirs.values():
                    for suffix in ["-center-crop.png", "-padded.png"]:
                        p = os.path.join(d, f"{fn_base}{suffix}")
                        if os.path.exists(p): os.remove(p)
            else:
                kept_urls.add(url)
        
        print(f"✅ Kept {len(kept_urls)} of {len(processed_urls)} images after blur filtering.")
        tune.orig_images = list(kept_urls) # Update tune object with final list

    # Generate the final, correctly structured metadata JSON
    print(f"\n⚙️  Generating multidatabackend.json ({version} Style)...")
    instance_prompt = _get_instance_prompt(tune)
    prompt_prefix = "A photo of " if 'style' not in tune.name else ""
    caption_strategy = tune.caption_strategy or "instanceprompt"

    data = [
        {"id": "text-embeds", "type": "local", "dataset_type": "text_embeds", "default": True, "cache_dir": f"{output_dir}/cache-text"},
    ]
    
    # Define the base datasets (face-crops and full-body)
    base_datasets_defs = [
        {
            "id": "dreambooth-data-face",
            "instance_data_dir": dirs['face_dir'],
            "cache_dir_vae": f"{output_dir}/cache-vae-flux-face",
            "instance_prompt": f"closeup {prompt_prefix}{instance_prompt}",
        },
        {
            "id": "dreambooth-data-training",
            "instance_data_dir": dirs['training_dir'],
            "cache_dir_vae": f"{output_dir}/cache-vae-flux-training",
            "instance_prompt": f"{prompt_prefix}{instance_prompt}",
        }
    ]

    # Add common attributes and add to the main data list
    final_base_datasets = []
    for dataset_def in base_datasets_defs:
        dataset_def.update({
            "type": "local", "dataset_type": "image",
            "resolution": tune.resolution, "crop": True, "crop_style": "center",
            "caption_strategy": caption_strategy, "only_instance_prompt": caption_strategy == 'instanceprompt',
            "metadata_backend": "json", "cache_file_suffix": "square",
        })
        final_base_datasets.append(dataset_def)
        _write_aspect_ratio_metadata(dataset_def['instance_data_dir'], tune.resolution, 'square')
    
    data.extend(final_base_datasets)
    
    # If segmentation is enabled, add separate, conditioned datasets
    if getattr(tune, "segmentation", False):
        for base_dataset in final_base_datasets:
            conditioned_copy = base_dataset.copy()
            conditioned_copy["id"] = f"{base_dataset['id']}-conditioned"
            conditioned_copy["cache_dir_vae"] = f"{base_dataset['cache_dir_vae']}-conditioned"
            conditioned_copy["conditioning_data"] = "dreambooth-conditioning"
            data.append(conditioned_copy)

        # Add the conditioning (mask) dataset itself
        data.append({
            "id": "dreambooth-conditioning", "type": "local", "dataset_type": "conditioning",
            "instance_data_dir": dirs['mask_dir'], "resolution": tune.resolution,
            "conditioning_type": "mask", "crop": True, "crop_style": "center",
            "caption_strategy": caption_strategy, "instance_prompt": instance_prompt,
            "only_instance_prompt": caption_strategy == 'instanceprompt', "cache_file_suffix": "square-mask"
        })
        _write_aspect_ratio_metadata(dirs['mask_dir'], tune.resolution, 'square-mask')

    multidatabackend_config = f"{output_dir}/multidatabackend.json"
    with open(multidatabackend_config, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ {version} processing complete. Metadata saved to {multidatabackend_config}")

    return multidatabackend_config

# ----------------------------------------------------------------------
# ─── MAIN DISPATCHER ──────────────────────────────────────────────────
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Data Preprocessing Orchestrator.")
    parser.add_argument("tune_id", type=str, help="The ID of the tune job to process.")
    parser.add_argument("--version", type=str, choices=['v1', 'v2', 'v3'], required=True, help="The version of the download script to emulate.")
    args = parser.parse_args()

    # MOCK a 'tune' object. In production, you would fetch this from a server.
    mock_tune = JsonObj({
        "id": args.tune_id, "resolution": 1024,
        "face_crop": True, "disable_face_crop": False, "segmentation": True,
        "only_face": True, "black_mask": True,
        "name": "woman", "token": "sks",
        "orig_images": [
             "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/over-shop-window.jpg",
             "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/over-cat.jpg",
        ]
    })

    output_dir = f"{EPHEMERAL_MODELS_DIR}/{mock_tune.id}-{args.version}-output"
    os.makedirs(output_dir, exist_ok=True)
    mock_tune.resolution = int(getattr(mock_tune, "resolution", 512)) # Ensure resolution is set

    if args.version == 'v1':
        run_v1_processing(mock_tune, output_dir)
    elif args.version in ['v2', 'v3']:
        run_v2_v3_processing(mock_tune, output_dir, args.version)
    else:
        raise ValueError("Invalid version specified.")