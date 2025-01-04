import re
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import os
import json
import requests
import shutil
import time
from pathlib import Path
import traceback
from functools import wraps
import rollbar
from filelock import FileLock, Timeout
import psutil

rollbar.init(
    access_token='c49dfbb91ec64c67bc6f06182be31150',
    environment='production',
    code_version='1.0.0',
)

REQUIRED_DISK_SPACE = 20 * 1024 * 1024 * 1024
MODELS_DIR = "/data/models"
CACHE_DIR = "/data/cache"
os.makedirs(MODELS_DIR, exist_ok=True)
EPHEMERAL_DIR = "/ephemeral-data"
EPHEMERAL_MODELS_DIR = "/ephemeral-data/models"
os.makedirs(EPHEMERAL_MODELS_DIR, exist_ok=True)
total_memory_in_GB = psutil.virtual_memory().total / (1024.0 ** 3)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
HUMAN_CLASS_NAMES = ['man', 'woman', 'boy', 'girl', 'child', 'baby', 'person']

device = "cuda"

class JsonObj(SimpleNamespace):

    def __init__(self, **dictionary):
        super().__init__()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, JsonObj(value))
            else:
                self.__setattr__(key, value)

    def __getattribute__(self, value):
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None

FLUX_INPAINT_MODEL_ID = '1832147'
DO_NOT_DELETE_MODELS = [
    '730158-sdxl1', # ZavyChroma
    '1121645-sdxl1', # ZavyChromaXL v5
    '1192646-sdxl1', # ZavyChromaXL v6.0
    '725831-sdxl1', # Crystal Clear XL CCXL
    '666678-sdxl1', # SDXL 1.0
    '678865-sd15', # RV5
    '690204-sd15', # RV5.1
    '627443-sd15', # RV3
    '444827-sd15', # RV2
    '317242-sd15', # Prism
    '379577-sd15', # Analog diffusion
    '1034743-sd15', # PicX

    # Adir app
    '950531-sd15', # Toon Babes for Adir
    '636337-sd15', # Pixar for Adir
    '883052-sd15', # Pixar for Adir

    # Flux
    '1504944-flux1',
    f'{FLUX_INPAINT_MODEL_ID}-flux1',

    'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    'AdaBins_nyu.pt',
    'dpt_large-midas-2f21e586.pt',
]

last_cleanup_time = 0
MAX_USED_SPACE = int(os.environ.get("MAX_USED_SPACE")) * 1024 * 1024 * 1024 if os.environ.get("MAX_USED_SPACE") else None
def cleanup_models(force_cleanup=False):
    timestamp_file = os.path.join(MODELS_DIR, "cleanup_timestamp.txt")

    # Check if enough time has passed since the last cleanup
    global last_cleanup_time
    if time.time() - last_cleanup_time < 1200:
        return
    last_cleanup_time = time.time()


    # Check if enough time has passed since the last cleanup - cross process
    if os.path.exists(timestamp_file) and not os.environ.get("FORCE_CLEANUP") and not force_cleanup:
        with open(timestamp_file, 'r') as file:
            try:
                last_cleanup = float(file.read().strip())
            except ValueError:
                # ValueError: could not convert string to float: ''
                return
            if time.time() - last_cleanup < 120:
                print("Less than 2 minutes since last cleanup")
                return

    lock_file = os.path.join(MODELS_DIR, "cleanup.lock")
    try:
        with FileLock(lock_file, timeout=0):
            for i in range(10):
                try:
                    listdir = [x for x in os.listdir(MODELS_DIR) if not x.endswith(".lock")]
                    listdir = sorted(listdir, key=lambda x: os.stat(os.path.join(MODELS_DIR, x)).st_atime)
                    break
                except FileNotFoundError as e:
                    # sometimes one process might delete a file while iterating the listdir
                    print(e)
                    time.sleep(1)
                    if i == 9:
                        raise e

            if MAX_USED_SPACE:
                # MAX_USED_SPACE is in GB
                dir_size = get_dir_size()
                free_space_needed = dir_size - MAX_USED_SPACE
                print(f'cleanup_models free free_space_needed={free_space_needed / 1024 / 1024 / 1024:.0f} GB dir_size={(dir_size / 1024 / 1024 / 1024):.0f} GB MAX_USED_SPACE={(MAX_USED_SPACE / 1024 / 1024 / 1024):.0f} total={len(listdir)}')
            else:
                # Since fsx reclaims space slowly, we need to calculate how much space we need to free up and not query disk_usage each time
                free_space = shutil.disk_usage(MODELS_DIR).free
                free_space_needed = REQUIRED_DISK_SPACE - free_space
                print(f'cleanup_models free space={(free_space / 1024 / 1024 / 1024):.1f} GB total={len(listdir)}')
            i = 0

            while i < 1000 and (free_space_needed > 0 or len(listdir) > 2500):
                i += 1
                dirname = listdir[0]
                dir_path = os.path.join(MODELS_DIR, dirname)

                if dirname.replace('.safetensors', '') in DO_NOT_DELETE_MODELS:
                    pass  # Skipping
                else:
                    size = None
                    if os.path.isfile(dir_path):
                        size = os.path.getsize(dir_path)
                        os.remove(dir_path)
                        shutil.rmtree(f"{MODELS_DIR}/{dirname}.lock", ignore_errors=True)
                        free_space_needed -= size
                    else:
                        if not os.path.exists(os.path.join(dir_path, "do_not_delete")):
                            size = get_dir_size(dir_path)
                            shutil.rmtree(dir_path)
                            shutil.rmtree(f"{MODELS_DIR}/{dirname}.lock", ignore_errors=True)
                            free_space_needed -= size
                    if size:
                        print(f"Deleted {i}, {dir_path}, {size / 1024 / 1024 / 1024:.1f} GB free_space_needed={free_space_needed / 1024 / 1024 / 1024:.1f} GB")

                listdir = listdir[1:]

            if i >= 100:
                rollbar.report_message("Cleaned max number of models but didn't reach free space", "warning")
            # At the end of the cleanup, update the timestamp
            with open(timestamp_file, 'w') as file:
                file.write(str(time.time()))

    except Timeout:
        print("Lock is taken, continuing without cleanup")

def get_dir_size(path='/data/models /data/cache /data/a1111-models /data/priors'):
    # avoid concurrent calls to dut - add some random number to the filename
    random_tempfilename = f"/tmp/dut_output_{time.time()}.txt"
    print(random_tempfilename)
    os.system(f"dut -b {path} | tail -n1 > {random_tempfilename}")
    with open(random_tempfilename, "r") as f:
        line = f.read()
        # parse bytes-used using regex
    # delete the temp file
    os.remove(random_tempfilename)
    return int(re.findall(r'(\d+)', line)[0])

class StaleDeploymentException(Exception):
    pass

def run(args, **kwargs):
    # if kwargs not empty, use subprocess.run
    if kwargs:
        return subprocess.run(args, check=True, **kwargs)

    args = [shlex.quote(arg) for arg in args]
    print(" ".join(args))
    exit_code = os.system(" ".join(args))
    if exit_code != 0:
        raise Exception(f"Command failed: {' '.join(args)}")

def tail(f, n, offset=0):
    proc = subprocess.Popen(['tail', '-n', str(n + offset), f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return lines

def run_with_output(args, **kwargs):
    args = [shlex.quote(arg) for arg in args]
    print(" ".join(args))

    # Use os.system to execute the command and redirect stdout to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        exit_code = os.system(f"{' '.join(args)} > {temp.name}")

    # Read the output from the temporary file and split it into lines
    tail_lines = tail(temp.name, 100)

    # Delete the temporary file
    os.unlink(temp.name)

    # Check the exit code
    if exit_code != 0:
        raise Exception(f"Command failed: {' '.join(args)}: {tail_lines}")

    # Return the last 100 lines of the output
    return tail_lines

def download_model_from_server(model_name: str, convert_xl_to_diffusers = True):
    cleanup_models()
    # Playground V2 has a different configuration - https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic
    if model_name == '1504944-flux1':
        model_path = f"{MODELS_DIR}/{model_name}"
        if not os.path.exists(model_path) or os.environ.get("FORCE_DOWNLOAD") or not os.path.exists(f"{model_path}/model_index.json"):
            from huggingface_hub import snapshot_download
            snapshot_download('black-forest-labs/FLUX.1-dev', ignore_patterns=['flux1-dev.safetensors', 'ae.safetensors', 'dev_grid.jpg', 'README.md', 'LICENSE.md', '.gitattributes'], local_dir=model_path)
            Path(f"{model_path}/do_not_delete").touch()
        return model_path
    if model_name == f'{FLUX_INPAINT_MODEL_ID}-flux1':
        model_path = f"{MODELS_DIR}/{model_name}"
        if not os.path.exists(model_path) or os.environ.get("FORCE_DOWNLOAD") or not os.path.exists(f"{model_path}/model_index.json"):
            from huggingface_hub import snapshot_download
            snapshot_download('black-forest-labs/FLUX.1-Fill-dev',
                              # ignore_patterns=['ae.safetensors', 'dev_grid.jpg', 'README.md', 'LICENSE.md', '.gitattributes'],
                              allow_patterns=['transformer/'],
                              local_dir=model_path, revision="refs/pr/4",
                              )
            Path(f"{model_path}/do_not_delete").touch()
        return model_path
    raise NotImplementedError(f"model_name={model_name}")


if os.environ.get("MOCK_SERVER", False):
    check_refresh = lambda _: True


if __name__ == "__main__":
    print("Starting")
    while True:
        cleanup_models()
        sleep_time = 60 * 60 * 4
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
    print("Done")
