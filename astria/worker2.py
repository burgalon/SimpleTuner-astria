import json
import re
import os
import shutil
import time
import sys
import threading
import uvicorn
import requests
import concurrent.futures
import base64
from pathlib import Path

import torch
import torch.distributed as dist

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your FastAPI app ---
from astria.preprocess_server import BASE_PREPROCESS_PORT, app as fastapi_app

from helpers.training.trainer import Trainer

from sig_listener import set_trainer_instance
from astria.dummy_training import create_dummy_data_config
from astria.train import download_dev2pro, create_prompt_library, parse_env_args, parse_args
from astria.download_training_local_api import run_v1_processing, run_v2_v3_processing
from astria_utils import run, run_with_output, MODELS_DIR, EPHEMERAL_MODELS_DIR, \
    download_model_from_server, JsonObj, cleanup_models, CUDA_VISIBLE_DEVICES, upload_to_sync, HUMAN_CLASS_NAMES
from cleanup_directory import cleanup_directory


from astria_tests.fixtures.worker_fixtures import (
    JOB_STR_1, JOB_STR_2, JOB_STR_3, JOB_STR_4, JOB_STR_5, JOB_STR_HAIR
)

MEGA_CACHE_PATH = "/data/cache/flux_transformer_mega_cache.bin"
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3

def is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0

def is_rank1() -> bool:
    return dist.is_initialized() and dist.get_rank() == 1


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

class Worker:
    def __init__(self):
        self.preprocess_server_ports = [BASE_PREPROCESS_PORT + i for i in range(NUM_GPUS)]

        # load config/warmup.json
        os.environ['SIMPLETUNER_CONFIG_BACKEND'] = 'json'
        os.environ['SIMPLETUNER_ENVIRONMENT'] = 'warmup'
        self.trainer = Trainer(
            keep_backbone_loaded=True,
            report_to='wandb' if os.environ.get('TRAIN_WANDB', False) else None,
            skip_unload_supporting_models=True,
            torch_compile_transformer=os.environ.get('TORCH_COMPILE_TRANSFORMER', False),
        )

        self._start_preprocessing_server()

        set_trainer_instance(self.trainer)
        os.environ['SIMPLETUNER_CONFIG_BACKEND'] = 'cmd'
        os.environ['SIMPLETUNER_ENVIRONMENT'] = ''
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/data/cache/torchinductor_cache'
        
        start = time.time()
        self.setup_trainer(dummy_config=True)
        self.run()
        self.trainer.cleanup_for_next_lora()
        print('Trainer setup and pre-warmed in', time.time() - start)
        
        if os.environ.get('TORCH_COMPILE_TRANSFORMER', False):
            if self.trainer.accelerator.is_main_process:
                if not self.load_mega_cache():
                    self.save_mega_cache()
            self.load_mega_cache()

    def _start_preprocessing_server(self):
        """Starts the FastAPI server in a background daemon thread for each worker process."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        port = BASE_PREPROCESS_PORT + rank

        def run_server():
            print(f"[GPU-{rank}] Starting preprocess server on port {port}...")
            uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)

    def load_mega_cache(self, pth=MEGA_CACHE_PATH):
        if Path(pth).exists():
            artifact_bytes = open(pth, "rb").read()
            torch.compiler.load_cache_artifacts(artifact_bytes)
            print(f"[mega-cache] Loaded cache from {pth}")
            return True
        else:
            print(f"[mega-cache] No cache found at {pth}, will be generated.")
            return False

    def save_mega_cache(self, pth=MEGA_CACHE_PATH):
        if self.trainer.accelerator.is_main_process:
            artifact_bytes, cache_info = torch.compiler.save_cache_artifacts()
            with open(pth, "wb") as f:
                f.write(artifact_bytes)
            print(f"[mega-cache] Saved cache to {pth}")

    def setup_trainer(self, job: str=JOB_STR_3, dummy_config=False):
        tune = json.loads(job, object_hook=lambda d: JsonObj(**d))

        cleanup_models()
        parse_args(tune)
        # Download base model flux from huggingface hub
        model_path = download_model_from_server(f"{tune.base_tune_id}-{tune.branch}")
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        cleanup_directory(EPHEMERAL_MODELS_DIR)
        cleanup_models()
        output_dir = f"{MODELS_DIR}/{tune.id}-{tune.branch}"
        if os.environ.get('RETRAIN') or tune.user_id == 2:
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        caption_strategy = tune.caption_strategy or "instanceprompt"

        # Working setup https://wandb.ai/astria/lora-training/runs/b94a195701ed0a7d7b53e6c9771c4388?nw=nwuserburgalonastria
        # A40, train_batch_4 resolution=1024 lora_rank=64 lora_alpha=64 optimizer=adamw learning_rate=1e-4 max_grad_norm=None flux_lora_target=Before it was committed lr_warmpup_steps=1
        # 16.6s/it 3k steps 13:40 hours memory

        # if tune.user_id == 2:
        if os.environ.get('TRAIN_WANDB') and not dummy_config:
            if is_rank0():
                print('Training runs are being reported to wandb')
            tune.report_to = "wandb"

        parse_env_args(tune)
        if os.environ.get('TR_DISABLE_FACE_CROP'):
            tune.face_crop = False

        if tune.report_to == 'wandb' and not tune.validation_steps:
            tune.validation_steps = 100
        elif not tune.validation_steps:
            tune.validation_steps = 50000

        steps = min(5000, int(tune.steps) if tune.steps else 2000)

        print(f"output_dir={output_dir} steps={steps}")
        self.trainer.accelerator.wait_for_everyone()

        print(f"[GPU-{self.trainer.accelerator.process_index}] Waiting for preprocessing to finish...")
        data_backend_config = None
        if self.trainer.accelerator.is_main_process:
            if not dummy_config:
                version = getattr(tune, 'preprocessing', '2') # Default to v2
                if version == '1':
                    data_backend_config = run_v1_processing(tune, output_dir, self.preprocess_server_ports)
                else: # v2 or v3
                    data_backend_config = run_v2_v3_processing(tune, output_dir, version, self.preprocess_server_ports)
            else:
                data_backend_config, _ = create_dummy_data_config(output_dir)

        # TODO
        resolution = 512
        data_backend_config = f"{output_dir}/multidatabackend.json"
        steps = min(5000, int(tune.steps) if tune.steps else 2000)

        if tune.train_batch:
            train_batch = int(tune.train_batch) // NUM_GPUS
            if train_batch > len(tune.orig_images):
                train_batch = len(tune.orig_images)
        else:
            train_batch = max(1, (min(min(4, len(tune.orig_images)), 4))) // NUM_GPUS

        if dummy_config:
            tune.steps = 1
            steps = 1

        print('Training with per GPU batch size of:', train_batch)
        print('Effective batch size:', train_batch * NUM_GPUS)

        # Launching
        # export CUDA_VISIBLE_DEVICES="0,1"
        # export NUM_GPUS="2"
        # TR_STEPS=105 MOCK_SERVER=1 accelerate launch --gpu_ids="$CUDA_VISIBLE_DEVICES" --num_machines=1 --num_processes="$NUM_GPUS" --mixed_precision=no astria/worker.py
        args = [
            # 'accelerate',
            # 'launch',
            # '--gpu_ids', CUDA_VISIBLE_DEVICES,
            # '--mixed_precision=no',
            # '--multi_gpu',
            # # *([f'--multi_gpu'] if NUM_GPUS > 1 else []),
            # # f'--num_processes={NUM_GPUS}',
            # '--num_machines=1',
            # # 2 GPUs
            # # 1.77it/s dynamo_backend=no
            # # 1.85it/s dynamo_backend=cudagraphs
            # # 2.8it/s dynamo_backend=inductor
            # # 1 GPU
            # # 1.04s/it dynamo_backend=no
            # # 1.80it/s dynamo_backend=inductor
            # '--dynamo_backend', 'inductor',
            # 'train.py',
            # '--base_model_default_dtype=fp32',
            '--model_type=lora',
            *(['--flux_guidance_mode', tune.flux_guidance_mode] if tune.flux_guidance_mode else []),
            '--pretrained_model_name_or_path', model_path,
            *([
                  '--pretrained_transformer_model_name_or_path', download_dev2pro(),
                  '--pretrained_transformer_subfolder', 'none',
              ] if tune.dev2pro else []),
            '--peft_model_precision=bf16', # or bf16; when using adamw_bf16 this should be bf16
            '--set_grads_to_none', # ?
            '--gradient_accumulation_steps', str(tune.gradient_accumulation_steps or 1),
            '--resume_from_checkpoint=latest',
            '--snr_gamma', str(tune.snr_gamma or 5),
            '--data_backend_config', data_backend_config,
            '--aspect_bucket_rounding=2',
            '--num_train_epochs=0',
            f'--max_train_steps={steps}',
            # '--metadata_update_interval=65', # ?
            # https://wandb.ai/astria/lora-training/runs/b94a195701ed0a7d7b53e6c9771c4388?nw=nwuserburgalonastria
            *([f'--max_grad_norm={tune.max_grad_norm}'] if tune.max_grad_norm else []),
            f'--optimizer={tune.optimizer or "adamw"}', # previously, adamw_bf16
            f'--lora_type', tune.lora_type or 'standard',
            '--init_lokr_norm', str(tune.init_lokr_norm or 1e-3),
            # "--lycoris_config=config/lycoris_config.json",
            f'--learning_rate={tune.learning_rate or 1e-4}',
            '--lr_scheduler', tune.lr_scheduler or 'constant_with_warmup',
            '--seed=42',
            '--lr_warmup_steps=10',
            '--output_dir', output_dir,
            # '--inference_scheduler_timestep_spacing=trailing', # defaults
            # '--training_scheduler_timestep_spacing=trailing', # defaults
            '--report_to', tune.report_to or 'none',
            # '--allow_tf32', # deprecated
            # '--mixed_precision=bf16',
            # *([f'--base_model_precision={os.environ.get("BASE_MODEL_PRECISION")}'] if os.environ.get("BASE_MODEL_PRECISION") else []),
            *([f'--base_model_precision={tune.base_model_precision}'] if tune.base_model_precision else []),
            # helps see that we're not destroying the priors
            # '--validation_disable_unconditional',
            # '--i_know_what_i_am_doing',
            '--keep_vae_loaded',
            # ["mmdit", "context", "all"]
            *([f'--flux_lora_target={tune.flux_lora_target}'] if tune.flux_lora_target else []),
            f'--lora_rank={tune.lora_rank or 64}',
            f'--lora_alpha={tune.lora_alpha or 64}',
            *(['--user_prompt_library', create_prompt_library(tune, output_dir)] if tune.report_to else []),
            '--model_family=flux',
            f'--train_batch_size={train_batch}',
            # '--max_workers=1',
            # '--read_batch_size=1',
            # '--write_batch_size=1',
            # '--override_dataset_config',
            '--caption_dropout_probability', str(tune.caption_dropout_probability if tune.caption_dropout_probability is not None else 0.1),
            *(['--use_ema'] if tune.use_ema else []),
            # '--ema_decay=0.99',
            # '--torch_num_threads=8',
            # '--image_processing_batch_size=32',
            # '--vae_batch_size=1',
            # '--validation_prompt="ohwx woman holding flowers, red sweater, studio photography, plain white background"',
            '--num_validation_images=1',
            '--validation_num_inference_steps=28',
            '--validation_seed=42',
            '--minimum_image_size=0',
            f'--resolution={resolution}',
            '--validation_resolution=1024x1024',
            '--resolution_type=pixel',
            '--checkpointing_steps', str(tune.checkpointing_steps or 1000),
            '--checkpoints_total_limit=10',
            '--validation_steps', str(tune.validation_steps) if tune.validation_steps else '5000',
            f'--tracker_run_name={tune.id}-{tune.branch}-{os.environ.get("TRACKER_NAME", timestamp)} {tune.title} {tune.args}',
            *(['--evaluation_type=face'] if tune.report_to and tune.face_crop else []),
            '--tracker_project_name=flux-lora',
            '--validation_guidance=3.5',
            '--validation_guidance_rescale=0.0',
            '--disable_benchmark',
            '--flux_schedule_auto_shift',
            '--flux_schedule_shift=0',
            # '--skip_file_discovery=aspect,metadata',
            *(['--prepend_instance_prompt'] if caption_strategy == "textfile" else []),
                            ]
        old_config = self.trainer.config
        self.trainer.parse_arguments(args)
        # iterate Namespace returned by parse_args and merge it with former
        for key, value in self.trainer.config.__dict__.items():
            setattr(old_config, key, value)
        print("data_backend_config:", self.trainer.config.data_backend_config)
        print("train_batch=", self.trainer.config.train_batch_size)
        self.trainer.config = old_config

    def run(self):
        self.trainer.run()
        self.trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    worker = Worker()
    start1 = time.time()
    worker.setup_trainer(JOB_STR_HAIR)
    worker.run()
    end1 = time.time()

    start2 = time.time()
    worker.trainer.cleanup_for_next_lora()
    worker.setup_trainer(JOB_STR_3)
    worker.run()

    if worker.trainer.accelerator.is_main_process:
        print('Ran training loop 1 in ', end1 - start1)
        print('Ran training loop 2 in ', time.time() - start2)