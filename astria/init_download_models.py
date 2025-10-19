import time
import os

from hinter_helper import annotator_ckpts_path
import torch
from diffusers import FluxControlNetModel
from filelock import FileLock, Timeout
from huggingface_hub import scan_cache_dir, snapshot_download
from torch.hub import download_url_to_file
from transformers import pipeline

from astria.seedvr2.download import ensure_seedvr2_7b_checkpoint, ensure_seedvr2_vae_checkpoint
from astria_utils import MODELS_DIR, CACHE_DIR, run, download_model_from_server, FLUX_INPAINT_MODEL_ID
from controlnet_constants import CONTROLNETS_DICT
from add_clut import CLUT_DICT
from pulid_pipeline.pulid_ext import PuLID

def get_cached_repos_dict():
    return dict((repo.repo_id, repo) for repo in scan_cache_dir().repos)

def download_hinters(cached_repos_dict):
    if "depth-anything/Depth-Anything-V2-Small-hf" not in cached_repos_dict:
        pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    os.makedirs(annotator_ckpts_path, exist_ok=True)
    urls = [
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/network-bsds500.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth",
        "https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/150_16_swin_l_oneformer_coco_100ep.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/250_16_swin_l_oneformer_ade20k_160k.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt",
        "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth",
        "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth",
        "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth",
        # "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
        # "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ]
    for url in urls:
        if not os.path.exists(os.path.join(annotator_ckpts_path, os.path.basename(url))):
            download_url_to_file(url, os.path.join(annotator_ckpts_path, os.path.basename(url)))

def _download_models():
    # download_model_from_server('1504944-flux1', False)
    snapshot_download('black-forest-labs/FLUX.1-dev', ignore_patterns=['flux1-dev.safetensors', 'ae.safetensors', 'dev_grid.jpg', 'README.md', 'LICENSE.md', '.gitattributes'], local_dir=f"{MODELS_DIR}/1504944-flux1")
    # HF_TOKEN=hf_********tNke huggingface-cli upload tuner /data/cache / --exclude "torchinductor/**" .cache *.log Wan-AI
    snapshot_download(repo_id="burgalon/tuner", local_dir=CACHE_DIR, local_dir_use_symlinks=False)

    # extract tar /data/cache/torchinductor.tar
    run(['tar', '-xf', f'{CACHE_DIR}/torchinductor.tar', '-C', CACHE_DIR])
    # for reference - tar command - tar -cf /data/cache/torchinductor.tar -C /data/cache torchinductor
    # now upload to huggingface - huggingface-cli upload burgalon/tuner /data/cache/torchinductor.tar

    # BiRefNet
    if not os.path.exists(f"{CACHE_DIR}/BiRefNet/swin_large_patch4_window12_384_22kto1k.pth"):
        print("Downloading BiRefNet model")
        snapshot_download(
            repo_id="ViperYX/BiRefNet",
            allow_patterns=[f"*swin_large_patch4_window12_384_22kto1k*"],
            local_dir=CACHE_DIR + ' BiRefNet',
            local_dir_use_symlinks=False,
        )

    print("Downloading PuLID models")
    PuLID.download_models(local_dir=f'{CACHE_DIR}/pulid', models_dir=CACHE_DIR)

    fns = [
        # ('https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth', CACHE_DIR),
        ('https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth', CACHE_DIR),
        ('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt', CACHE_DIR),
        # PuLID parsenet
        ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', CACHE_DIR),
        ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', CACHE_DIR),
        ('https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth', CACHE_DIR),

    ]
    for url, path in fns:
        if '/blob/' in url:
            raise ValueError(f"URL {url} is a blob URL, please use the 'raw' URL")
        target_fn = path if '.' in path else f"{path}/{os.path.basename(url)}"
        base_dir = os.path.dirname(target_fn)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(target_fn):
            print(f"Downloading {url} to {target_fn}")
            download_url_to_file(url, target_fn)

def _download_models_secondary():
    # wait for download_model.lock to release
    with FileLock(f"{MODELS_DIR}/download_model.lock", timeout=0):
        print("Waiting for primary download to release download_model.lock")
        pass

    # download_model_from_server(f'{FLUX_INPAINT_MODEL_ID}-flux1', False)
    snapshot_download('black-forest-labs/FLUX.1-Fill-dev',
                      allow_patterns=['transformer/'],
                      local_dir=f'{MODELS_DIR}/{FLUX_INPAINT_MODEL_ID}-flux1',
                      revision="refs/pr/4",
                      )

    print("Downloading models in secondary process")
    download_model_from_server(f"3063697-flux1") # Krea

    # seedvr2 7b
    ensure_seedvr2_7b_checkpoint(target_dir=str(CACHE_DIR))
    ensure_seedvr2_vae_checkpoint(target_dir=str(CACHE_DIR))

    snapshot_download(repo_id="HCMUE-Research/SAM-vit-h", local_dir=CACHE_DIR, local_dir_use_symlinks=False, allow_patterns=["*.pth"])
    os.makedirs("/data/cache/HaldCLUT", exist_ok=True)

    cached_repos_dict = get_cached_repos_dict()

    for dict in CONTROLNETS_DICT.values():
        for key, model_name in dict.items():
            if (isinstance(model_name, str) and model_name not in cached_repos_dict):
                print(f"Downloading {model_name}")
                FluxControlNetModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    # local_files_only=True,
                )
    download_hinters(cached_repos_dict)
    download_model_from_server(f"3086296-qwen-image-1") # QWEN
    download_model_from_server(f"3123913-qwen-edit-1") # QWEN-Edit


def _download_models_secondary_with_retry():
    for i in range(10):
        try:
            _download_models_secondary()
            break
        except Exception as e:
            print(f"Download secondary failed: {e}. Sleeping for 10 seconds")
            time.sleep(10)


def download_models_with_lock():
    try:
        with FileLock(f"{MODELS_DIR}/download_model.lock", timeout=0):
            for i in range(10):
                try:
                    _download_models()
                    break
                except Exception as e:
                    print(f"Download failed: {e}. Sleeping for 10 seconds")
                    time.sleep(10)

    except Timeout:
        print("Another process is downloading models, skipping download")



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        download_models_with_lock()
    else:
        _download_models_secondary_with_retry()
