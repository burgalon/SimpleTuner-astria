import json
import os
import traceback

import requests

from astria_utils import CUDA_VISIBLE_DEVICES, JsonObj, StaleDeploymentException
from request_session import session

DOMAIN = os.environ.get("DOMAIN", "http://sdbooth.herokuapp.com/")
IS_INPUT_VIDEO = os.environ.get("IS_INPUT_VIDEO", "false") == "true"
BRANCHES = os.environ.get("BRANCHES", "branches[]=flux1")
IMAGE = '&image=ai-toolkit' if os.path.exists('flux_train_ui.py') else ''
BACKEND_VERSION = 0

FASHN_API_KEY = '462ab748-3f14-441b-b06b-aa295cb2ae4e:0e3d97023da6a18d69105588305d487c'

HEADERS = {
    "DEPLOY_TS": os.environ.get("DEPLOY_TS", ""),
    "RUNPOD_POD_ID": (f"akash-{os.environ.get('AKASH_DEPLOYMENT_SEQUENCE')}" if os.environ.get('AKASH_DEPLOYMENT_SEQUENCE') else os.environ.get("RUNPOD_POD_ID", "")) + "-" + CUDA_VISIBLE_DEVICES,
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
    if id is not None:
        print(f"Requesting job for tune {id}")
        response = requests.post(f"{DOMAIN}tunes/train?id={id}", headers=HEADERS)
    else:
        response = requests.post(f"{DOMAIN}tunes/train?{BRANCHES}{IMAGE}", headers=HEADERS)
    ret =  json.loads(response.text, object_hook=lambda d: JsonObj(**d))
    if ret.message and "Stale deployment" in ret.message:
        raise StaleDeploymentException()
    return ret

def request_infer_job_from_server(tune_id: str = None, is_video: bool = False, id: str = None):
    is_video_qs = "&is_video=true" if is_video or IS_INPUT_VIDEO else ""
    is_input_video_qs = "&is_input_video=true" if IS_INPUT_VIDEO else ""
    backend_version_qs = "&backend_version=" + str(BACKEND_VERSION)
    try:
        if id is not None:
            print(f"Requesting job for prompt {id}")
            response = requests.post(f"{DOMAIN}prompts/train_batch?id={id}{backend_version_qs}")
        elif tune_id is None:
            response = requests.post(f"{DOMAIN}prompts/train_batch?{BRANCHES}{is_video_qs}{is_input_video_qs}{backend_version_qs}", headers=HEADERS)
        else:
            response = requests.post(f"{DOMAIN}prompts/train_batch?tune_id={tune_id}{is_video_qs}{is_input_video_qs}{backend_version_qs}", headers=HEADERS)
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
