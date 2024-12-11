import os

from hinter_helper import annotator_ckpts_path
import torch
from diffusers import FluxControlNetModel
from filelock import FileLock, Timeout
from huggingface_hub import scan_cache_dir
from torch.hub import download_url_to_file
from transformers import pipeline

from astria_utils import MODELS_DIR, CACHE_DIR, run, download_model_from_server
from controlnet_constants import CONTROLNETS_DICT
from add_clut import CLUT_DICT


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
    download_model_from_server('1504944-flux1', False)
    cached_repos_dict = get_cached_repos_dict()
    # TODO
    # download_path: /app/models/insightface/models/buffalo_l
    #  Downloading /app/models/insightface/models/buffalo_l.zip from https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip...

    fns = [
        # ('https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth', CACHE_DIR),
        ('https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth', CACHE_DIR),
        ('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt', CACHE_DIR),

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

    os.makedirs("/data/cache/HaldCLUT", exist_ok=True)
    for filename in [*CLUT_DICT.values()]:
        if not os.path.exists(f"/data/cache/{filename}"):
            run(['aws', 's3', 'cp', f's3://astria-model-repo/cache/{filename}', f'/data/cache/{filename}'])

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



def download_models_with_lock():
    try:
        with FileLock(f"{MODELS_DIR}/download_model.lock", timeout=0):
            _download_models()
    except Timeout:
        print("Another process is downloading models, skipping download")



if __name__ == "__main__":
    download_models_with_lock()
