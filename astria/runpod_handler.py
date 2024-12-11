import json
import os
import traceback

import runpod
import torch

from train import poll_train
from infer import InferPipeline
import infer
from astria_utils import run, JsonObj, MODELS_DIR
from sig_listener import TerminateException, handle_sigterm

if not os.environ.get('DEBUG'):
    run(["sh", "astria/set_runpod_mounts.sh"])

pipeline = InferPipeline()
pipeline.init_pipe(MODELS_DIR + "/1504944-flux1")

def handler(job):
    global pipeline
    data = job["input"]
    data = json.loads(json.dumps(data), object_hook=lambda d: JsonObj(**d))

    try:
        if data.poll_train:
            infer.pipe = None
            processed_jobs = 1
            while processed_jobs>0:
                pipeline.reset()
                processed_jobs = poll_train()
            return { "success": True, "processed_jobs": processed_jobs}

        elif data.poll_infer:
            processed_jobs = 1
            while processed_jobs>0:
                processed_jobs = pipeline.poll_infer()
            return { "success": True, "processed_jobs": processed_jobs}

        else:
            raise Exception("Invalid job")
    except (TerminateException, torch.cuda.OutOfMemoryError):
        return { "refresh_worker": True, "error": traceback.format_exc()}

runpod.serverless.start({ "handler": handler})
