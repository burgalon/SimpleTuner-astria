import subprocess

import json
import os
import traceback

import requests

from astria_utils import CUDA_VISIBLE_DEVICES, JsonObj, StaleDeploymentException, CACHE_DIR, MODELS_DIR
from request_session import session

DOMAIN = os.environ.get("DOMAIN", "http://sdbooth.herokuapp.com/")
BRANCHES = os.environ.get("BRANCHES", "branches[]=flux1")
IMAGE = '&image=ai-toolkit' if os.path.exists('flux_train_ui.py') else ''
BACKEND_VERSION = 0
# shell  from nvidia-smi since torch is limited to CUDA_VISIBLE_DEVICES, do NOT use torch.cuda.device_count()
# NOTE NUM_GPUS is a string
NUM_GPUS = subprocess.run("nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1", shell=True, check=True, stdout=subprocess.PIPE).stdout.decode().strip()
FASHN_API_KEY = '462ab748-3f14-441b-b06b-aa295cb2ae4e:0e3d97023da6a18d69105588305d487c'
BAKE_API_KEY = '3477ba4e-b14f-4309-a4b2-3d4e5dce0587'
# (connect timeout, read timeout)
REQUEST_TIMEOUT = (10, 20)

HEADERS = {
    "DEPLOY_TS": os.environ.get("DEPLOY_TS", ""),
    "RUNPOD_POD_ID": (f"akash-{os.environ.get('AKASH_DEPLOYMENT_SEQUENCE')}" if os.environ.get('AKASH_DEPLOYMENT_SEQUENCE') else os.environ.get("RUNPOD_POD_ID", "")) + "-" + CUDA_VISIBLE_DEVICES,
    "AKASH_CLUSTER_PUBLIC_HOSTNAME": os.environ.get("AKASH_CLUSTER_PUBLIC_HOSTNAME", ""),
    "NUM_GPUS": str(NUM_GPUS),
    "SERVICE": os.environ.get("HOSTNAME", ""),
    "Authorization": f"Bearer {os.environ.get('ASTRIA_API_KEY', 'NONE')}",
}
print(f"headers={HEADERS}")
print(f"IMAGE={IMAGE}")

def report_tune_job_failure(tune: JsonObj, fail_message: str):
    session.post(f"{DOMAIN}tunes/{tune.id}/failed", json={"fail_message": fail_message})

def report_infer_job_failure(prompt: JsonObj, trace_str: str):
    try:
        print(f"Reporting job failure prompt {prompt.id}")
        response = session.post(f"{DOMAIN}prompts/{prompt.id}/failed", json={"fail_message": trace_str}, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        traceback.print_exc()

def request_tune_job_from_server(id = None):
    if allow_polling_qwen():
        qwen_qs = "&branches[]=qwen-image-1"
    else:
        qwen_qs = ""

    if id is not None:
        print(f"Requesting job for tune {id}")
        response = requests.post(f"{DOMAIN}tunes/train?id={id}", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    else:
        response = requests.post(f"{DOMAIN}tunes/train?{BRANCHES}{qwen_qs}{IMAGE}", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    ret =  json.loads(response.text, object_hook=lambda d: JsonObj(**d))
    if ret.message and "Stale deployment" in ret.message:
        raise StaleDeploymentException()
    return ret

def allow_polling_qwen():
    return os.path.exists(f"{MODELS_DIR}/3086296-qwen-image-1")

def allow_polling_qwen_edit():
    return os.path.exists(f"{MODELS_DIR}/3123913-qwen-edit-1")

def request_infer_job_from_server(tune_id: str = None, is_video: bool = False, id: str = None):
    qwen_qs = "&branches[]=qwen-image-1" if allow_polling_qwen() else ""
    qwen_edit_qs = "&branches[]=qwen-edit-1" if allow_polling_qwen_edit() else ""

    backend_version_qs = "&backend_version=" + str(BACKEND_VERSION)
    try:
        if id is not None:
            print(f"Requesting job for prompt {id}")
            response = requests.post(f"{DOMAIN}prompts/train_batch?id={id}{backend_version_qs}", timeout=REQUEST_TIMEOUT)
        elif tune_id is None:
            response = requests.post(f"{DOMAIN}prompts/train_batch?{BRANCHES}{qwen_qs}{qwen_edit_qs}{backend_version_qs}", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.post(f"{DOMAIN}prompts/train_batch?tune_id={tune_id}{qwen_qs}{qwen_edit_qs}{backend_version_qs}", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        ret =  json.loads(response.text, object_hook=lambda d: JsonObj(**d))
        if ret.message and "Stale deployment" in ret.message:
            raise StaleDeploymentException()
        return ret
    except requests.exceptions.RequestException as e:
        print(f"Failed to request job: {e}")
        return JsonObj()

def server_tune_done(tune: JsonObj):
    session.post(f"{DOMAIN}/tunes/{tune.id}/done")

if __name__ == "__main__":
    # For testing purposes, we can run this script directly
    print(f"Headers: {HEADERS}")
    response = request_infer_job_from_server()
    print(f"Response from server: {response}")

