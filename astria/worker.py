import json
import re
import os
import shutil
import time
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astria.train import download_dev2pro, create_prompt_library, parse_env_args, parse_args
from astria_utils import run, run_with_output, MODELS_DIR, EPHEMERAL_MODELS_DIR, \
    download_model_from_server, JsonObj, cleanup_models, CUDA_VISIBLE_DEVICES, upload_to_sync
from cleanup_directory import cleanup_directory
from download_training_v1 import create_data_config_v1
from download_training_v2 import create_data_config_v2
from download_training_v3 import create_data_config_v3
from helpers.training.trainer import Trainer
NUM_GPUS = torch.cuda.device_count()
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3



def extract_learning_rate(job_json: str) -> float|None:
    """
    Pull the `learning_rate` value from a job-spec JSON string.

    Parameters
    ----------
    job_json : str
        The raw JSON text shown in your example.

    Returns
    -------
    float | None
        The numeric learning-rate if it was present, otherwise ``None``.
    """
    # 1) Parse the JSON text into a Python dict
    job: dict = json.loads(job_json)

    # 2) Grab the free-form "args" field, if any
    args: str | None = job.get("args")
    if not args:
        return None

    # 3) Look for  learning_rate=...  (supports scientific notation, e.g. 5e-4)
    m = re.search(r"\blearning_rate\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", args)
    if not m:
        return None

    # 4) Convert to float and return
    return float(m.group(1))


class Worker:
    def __init__(self):
        # load config/warmup.json
        os.environ['SIMPLETUNER_CONFIG_BACKEND'] = 'json'
        os.environ['SIMPLETUNER_ENVIRONMENT'] = 'warmup'
        self.trainer = Trainer(keep_backbone_loaded=True)
        os.environ['SIMPLETUNER_CONFIG_BACKEND'] = 'cmd'
        os.environ['SIMPLETUNER_ENVIRONMENT'] = ''
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/data/cache/torchinductor_cache'
        os.environ['TORCH_COMPILE_DEBUG'] = "1"

    def setup_trainer(self, job: str):
        tune =  json.loads(job, object_hook=lambda d: JsonObj(**d))

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
        if os.environ.get('TRAIN_WANDB'):
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

        if tune.preprocessing=='3':
            print("Using preprocessing v3")
            data_backend_config, resolution = create_data_config_v3(tune, output_dir)
        elif tune.preprocessing=='2':
            print("Using preprocessing v2")
            data_backend_config, resolution = create_data_config_v2(tune, output_dir)
        else:
            print("Using preprocessing v1")
            data_backend_config, resolution = create_data_config_v1(tune, output_dir)


        # TODO
        resolution = 512
        # data_backend_config = multidatabackend_config = f"{output_dir}/multidatabackend.json"
        steps = min(5000, int(tune.steps) if tune.steps else 2000)


        if tune.train_batch:
            train_batch = int(tune.train_batch) // NUM_GPUS
            if train_batch > len(tune.orig_images):
                train_batch = len(tune.orig_images)
        else:
            train_batch = max(1, (min(min(4, len(tune.orig_images)), 4))) // NUM_GPUS

        # Launching
        # export CUDA_VISIBLE_DEVICES="0"
        # export NUM_GPUS="1"
        # TR_STEPS=10 MOCK_SERVER=1 accelerate launch --gpu_ids="$CUDA_VISIBLE_DEVICES" --num_machines=1 --num_processes="$NUM_GPUS" --mixed_precision=no --dynamo_backend="inductor" astria/worker.py
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
            '--resolution_type=pixel_area',
            '--checkpointing_steps', str(tune.checkpointing_steps or 1000),
            '--checkpoints_total_limit=10',
            '--validation_steps=100', # str(tune.validation_steps) if tune.validation_steps else '5000',
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


# PYTHONPATH=$PWD accelerate launch --gpu_ids 0 --mixed_precision=no --num_processes=1 --num_machines=1 --dynamo_backend=no  astria/worker.py
if __name__ == "__main__":
    print('hello')
    job_str = '{"id":2002368,"name":"man","created_at":"2025-01-05T08:58:31.696Z","updated_at":"2025-05-25T14:26:26.521Z","user_id":2,"trained_at":"2025-01-05T09:09:14.690Z","started_training_at":"2025-05-25T14:26:26.520Z","steps":405,"title":"Alon portrait","branch":"flux1","callback":null,"process_ip":"akash-wqtj5-2,3","trials":31,"num_prompts":0,"is_api":false,"base_tune_id":1504944,"token":"ohwx","args":"preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true","cost":null,"expires_at":"2025-06-04T09:09:14.690Z","emailed_notice":false,"public_at":null,"face_crop":true,"checkpoint_deleted":false,"checkpoint_deleted_at":null,"failed_at":null,"model_type":"lora","sha256":null,"model_url":null,"description_url":null,"cost_mc":216000,"training_face_correct":false,"eta":"2025-01-05T09:08:36.253Z","base_pack_id":260,"characteristics":{"age":"30 yo","ethnicity":"hispanic","eye_color":"brown eyes","facial_hair":"","glasses":"","hair_color":"black hair","hair_length":"short hair","hair_style":"bald","headcover":"","is_bald":"bald","name":"man"},"prompts_callback":null,"auto_extend":false,"orig_images":["https://sdbooth2-production.s3.amazonaws.com/ygesi07jlrw2nk31pq3tfu5vztgf","https://sdbooth2-production.s3.amazonaws.com/viatl4d53mu1ohsyymfu3hb6y74z","https://sdbooth2-production.s3.amazonaws.com/2mkj9khd3n2gm6usn0ppcnxxnyw6","https://sdbooth2-production.s3.amazonaws.com/2o99s9pxtny5ohrx0s1zp2wakha3","https://sdbooth2-production.s3.amazonaws.com/kjn9jgzchm4sl3uj1mo2zh2zv72f","https://sdbooth2-production.s3.amazonaws.com/3xfxajtel9jwq1al1fe8mey5rh17","https://sdbooth2-production.s3.amazonaws.com/4bp5gnc52qzbo2pj2nzwondpekui"],"file_names":[{"filename":"2xuxve6e6bh50w6sh97avcurxkeb.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/ygesi07jlrw2nk31pq3tfu5vztgf"},{"filename":"AA991103-E7DE-47C1-ABAB-82E794C150BF.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/viatl4d53mu1ohsyymfu3hb6y74z"},{"filename":"66gr44co9wa17nis5l5dnt3mg7gl.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/2mkj9khd3n2gm6usn0ppcnxxnyw6"},{"filename":"AA991103-E7DE-47C1-ABAB-82E794C150BF.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/2o99s9pxtny5ohrx0s1zp2wakha3"},{"filename":"IMG_8003.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/kjn9jgzchm4sl3uj1mo2zh2zv72f"},{"filename":"IMG_7988.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/3xfxajtel9jwq1al1fe8mey5rh17"},{"filename":"IMG_7659.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/4bp5gnc52qzbo2pj2nzwondpekui"}],"resolution":null,"user":{"create_ckpt":false,"backend_version":null},"prompts":[]}'
    worker = Worker()
    start = time.time()
    worker.setup_trainer(job_str)
    worker.run()
    print('Ran training loop 1 in ', time.time() - start)

    job_str = json.dumps({
        "id": 1858416,
        "name": "woman",
        "created_at": "2024-12-03T08:40:18.994Z",
        "updated_at": "2024-12-11T12:27:23.064Z",
        "user_id": 2,
        "trained_at": "2024-12-06T09:36:00.000Z",
        "started_training_at": "2024-12-11T12:27:23.063Z",
        "steps": 405,
        "title": "irit preset=portrait",
        "branch": "flux1",
        "callback": None,
        "process_ip": "akash-19180702-0",
        "trials": 19,
        "num_prompts": 0,
        "is_api": False,
        "base_tune_id": 1504944,
        "token": "ohwx",
        "args": "preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true",
        "cost": None,
        "expires_at": "2025-01-05T09:36:00.000Z",
        "emailed_notice": False,
        "public_at": None,
        "face_crop": True,
        "checkpoint_deleted": False,
        "checkpoint_deleted_at": None,
        "failed_at": "2024-12-03T10:20:01.543Z",
        "model_type": "lora",
        "sha256": "",
        "model_url": "",
        "description_url": "",
        "cost_mc": 0,
        "training_face_correct": False,
        "eta": "2024-12-06T10:00:55.196Z",
        "base_pack_id": None,
        "characteristics": None,
        "prompts_callback": None,
        "orig_images": [
        "https://sdbooth2-production.s3.amazonaws.com/jwaiky6g9b0ryqadm9gryz6wkykg",
        "https://sdbooth2-production.s3.amazonaws.com/uhtm480bwovnt4yi92eswipcjwqj",
        "https://sdbooth2-production.s3.amazonaws.com/hn2s2c5y461a5ve6y1266xgcc04c",
        "https://sdbooth2-production.s3.amazonaws.com/4skf7x147wrf9hjp61t8nvmrn9kj",
        "https://sdbooth2-production.s3.amazonaws.com/7jfx87xfrsl9hpp1jiiw3i2l440s",
        "https://sdbooth2-production.s3.amazonaws.com/37ebu983vz9kyeecjsg82f3wj23c",
        "https://sdbooth2-production.s3.amazonaws.com/cnt2pm2k4lg4933uy1w90fdr2jth",
        "https://sdbooth2-production.s3.amazonaws.com/6weh50yrdjly9utdwekemdmupocq",
        "https://sdbooth2-production.s3.amazonaws.com/bj8w0bwbrurt5cqzhsp99x27ffjc",
        "https://sdbooth2-production.s3.amazonaws.com/nvtmizkfjqh1hyfrgsjwoqq09zle",
        "https://sdbooth2-production.s3.amazonaws.com/4xvtx0n6529m6t8qhbzaz9xmekul",
        "https://sdbooth2-production.s3.amazonaws.com/602vsmswolbe9k2xde1if8uc0is4",
        "https://sdbooth2-production.s3.amazonaws.com/jn5ca51yn64u8mpx0eieqyeh3kid",
        "https://sdbooth2-production.s3.amazonaws.com/fiwba1158oocltizejyff86duhn9",
        "https://sdbooth2-production.s3.amazonaws.com/oomcupi58rxezfthrcx5d3oljj8x"
        ],
        "file_names": [
        {
            "filename": "ohwx (5).png",
            "url": "https://sdbooth2-production.s3.amazonaws.com/jwaiky6g9b0ryqadm9gryz6wkykg"
        },
        {
            "filename": "ohwx (15).JPG",
            "url": "https://sdbooth2-production.s3.amazonaws.com/uhtm480bwovnt4yi92eswipcjwqj"
        },
        {
            "filename": "ohwx (8).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/hn2s2c5y461a5ve6y1266xgcc04c"
        },
        {
            "filename": "ohwx (6).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/4skf7x147wrf9hjp61t8nvmrn9kj"
        },
        {
            "filename": "ohwx (7).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/7jfx87xfrsl9hpp1jiiw3i2l440s"
        },
        {
            "filename": "ohwx (1).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/37ebu983vz9kyeecjsg82f3wj23c"
        },
        {
            "filename": "ohwx (9).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/cnt2pm2k4lg4933uy1w90fdr2jth"
        },
        {
            "filename": "ohwx (14).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/6weh50yrdjly9utdwekemdmupocq"
        },
        {
            "filename": "ohwx (12).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/bj8w0bwbrurt5cqzhsp99x27ffjc"
        },
        {
            "filename": "ohwx (11).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/nvtmizkfjqh1hyfrgsjwoqq09zle"
        },
        {
            "filename": "ohwx (3).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/4xvtx0n6529m6t8qhbzaz9xmekul"
        },
        {
            "filename": "ohwx (10).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/602vsmswolbe9k2xde1if8uc0is4"
        },
        {
            "filename": "ohwx (4).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/jn5ca51yn64u8mpx0eieqyeh3kid"
        },
        {
            "filename": "ohwx (2).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/fiwba1158oocltizejyff86duhn9"
        },
        {
            "filename": "ohwx (13).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/oomcupi58rxezfthrcx5d3oljj8x"
        }
        ],
        "resolution": None,
        "user": {
        "create_ckpt": False,
        "backend_version": None
        },
        "prompts": []
    })
    start = time.time()
    worker.trainer.cleanup_for_next_lora(new_lr=extract_learning_rate(job_str))
    worker.setup_trainer(job_str)
    worker.run()
    print('Ran training loop 2 in ', time.time() - start)
