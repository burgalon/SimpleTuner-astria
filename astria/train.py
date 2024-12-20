import glob
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

from astria_utils import run, run_with_output, MODELS_DIR, EPHEMERAL_MODELS_DIR, \
    download_model_from_server, JsonObj, cleanup_models, CUDA_VISIBLE_DEVICES

if os.environ.get('MOCK_SERVER'):
    from astria_mock_server import request_tune_job_from_server, server_tune_done, report_tune_job_failure
else:
    from astria_server import request_tune_job_from_server, server_tune_done, report_tune_job_failure

from cleanup_directory import cleanup_directory
from sig_listener import is_terminated
from download_training_v1 import create_data_config_v1
from download_training_v2 import create_data_config_v2


def poll_train() -> int:
    tune = request_tune_job_from_server()
    if tune.id is None:
        return 0
    train(tune)
    return 1

def create_prompt_library(tune: JsonObj, output_dir: str):
    if tune.name == 'man':
        data = {
            "token_name": f"A detailed, high-quality photo of 25-year old black-haired Indian {tune.token} {tune.name} wearing glasses with short hair, wearing a beige-color blazer. The ohwx man has a muscular figure, is making eye-contact with the camera and striking a natural pose. The photo is taken from the chest up, in a bright and modern office building, using a shallow depth of field and bright, natural lighting to focus attention on the subject",
            # "flowers": f"{tune.name} holding flowers, red sweater, studio photography, plain white background",
            # "ohwx_rembrandt": f"a portrait of {tune.token} {tune.name} in the style of Rembrandt",
            # "rembrandt": f"a portrait of {tune.name} in the style of Rembrandt",
        }
    elif tune.name == 'girl':
        data = {
            "token_name": f"A photo of {tune.token} {tune.name}",
            "mushroom": f"boring bad quality snapchat photo circa 2015 of {tune.token} {tune.name} tinkerbell , green big dress,   translucent wings, golden dust around the air, sitting on top of massive 10feet mushroom  in forest, with a focus face slightly blurred, with digital noise, slightly pale yellow colour tone, looks like 2010 photo quality",
        }
    elif tune.name == 'woman':
        data = {
            "token_name": f"{tune.token} {tune.name}",
            "photo_token_name": f"A photo of {tune.token} {tune.name}",
            "flowers": f"{tune.token} {tune.name} holding flowers, white dress, black background",
            "suit": f"A {tune.token} {tune.name} with an intelligent, visionary gaze wearing a crisp, sky-blue blazer and coordinating slacks, presented in a high-resolution, venture capital-themed portrait.",
            "pearl": f"A {tune.token} {tune.name} with an approachable, friendly demeanor dressed in a soft, pearl-colored blouse and tailored black trousers, featured in a well-lit, business-oriented portrait.",
        }
    else:
        data = {
            "token_name": f"A photo of {tune.token} {tune.name}",
        }

    fn = f"{output_dir}/prompt_library.json"
    with open(fn, "w") as f:
        json.dump(data, f, indent=4)
    return fn


def caption(tune: JsonObj, multidatabackend_config: str, instance_data_dir: str):
    # if os.environ.get('SKIP_DOWNLOAD'):
    #     return

    # https://github.com/replicate/flux-fine-tuner/blob/main/caption.py#L49C1-L59C12
    prompt = """
    Write a four sentence caption for this image. In the first sentence describe the style and type (painting, photo, etc) of the image. Describe in the remaining sentences the contents and composition of the image. Only use language that would be used to prompt a text to image model. Do not include usage. Comma separate keywords rather than using "or". Precise composition is important. Avoid phrases like "conveys a sense of" and "capturing the", just use the terms themselves.
    
    Good examples are:
    
    "Photo of an alien woman with a glowing halo standing on top of a mountain, wearing a white robe and silver mask in the futuristic style with futuristic design, sky background, soft lighting, dynamic pose, a sense of future technology, a science fiction movie scene rendered in the Unreal Engine."
    
    "A scene from the cartoon series Masters of the Universe depicts Man-At-Arms wearing a gray helmet and gray armor with red gloves. He is holding an iron bar above his head while looking down on Orko, a pink blob character. Orko is sitting behind Man-At-Arms facing left on a chair. Both characters are standing near each other, with Orko inside a yellow chestplate over a blue shirt and black pants. The scene is drawn in the style of the Masters of the Universe cartoon series."
    
    "An emoji, digital illustration, playful, whimsical. A cartoon zombie character with green skin and tattered clothes reaches forward with two hands, they have green skin, messy hair, an open mouth and gaping teeth, one eye is half closed."
    """.strip()

    # # just one turn, always prepend image token
    # # inp = DEFAULT_IMAGE_TOKEN + "\n"
    # inp += PROMPT


    # This seems to mess the result
    # instance_prompt = get_instance_prompt(tune)
    # if tune.name == 'style':
    #     prompt += f"\n\nYou must end the caption with '{instance_prompt}'."
    # else:
    #     prompt += f"\n\nYou must start the caption with '{instance_prompt} '. "

    prompt += f"\n\nYou must start the caption with 'Photo of '. "

    run([
        "python3",
        "toolkit/captioning/caption_with_cogvlm.py",
        "--multidatabackend_config", multidatabackend_config,
        "--input_dir", instance_data_dir,
        "--caption_strategy=text",
        "--model_path", "THUDM/cogvlm-chat-hf",
        "--query", prompt,
    ])

    # Iterate over all TXT files and modify captions
    for txt_file in glob.glob(f"{instance_data_dir}/*.txt"):
        # created by caption_with_cogvlm.py
        if txt_file.endswith("processed_files.txt"):
            continue
        with open(txt_file, 'r') as f:
            content = f.read()

        # if tune.name not in content:
        #     content = f"{tune.name}, {content}"
        # modified_content = content.replace(tune.name, get_instance_prompt(tune))
        # remove any special tokens like <|end_of_text|> using regex, anything in between < and >
        # modified_content = re.sub(r'<[^>]*>', '', modified_content)
        modified_content = content
        with open(txt_file, 'w') as f:
            f.write(modified_content)
        print(f"Modified {txt_file}: {modified_content}")

def parse_args(tune: JsonObj):
    if tune.args:
        # split data.args string of the form key=value
        for arg in tune.args.strip().split(" "):
            key, value = arg.split("=", 1)
            setattr(tune, key, value)

def download_dev2pro():
    model_path = f"{MODELS_DIR}/dev2pro"
    if not os.path.exists('dev2pro'):
        from huggingface_hub import snapshot_download
        snapshot_download('ashen0209/Flux-Dev2Pro', local_dir=model_path)
        Path(f"{model_path}/do_not_delete").touch()
    return model_path


def train_no_catch(tune: JsonObj):
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

    # Working setup https://wandb.ai/astria/lora-training/runs/b94a195701ed0a7d7b53e6c9771c4388?nw=nwuserburgalonastria
    # A40, train_batch_4 resolution=1024 lora_rank=64 lora_alpha=64 optimizer=adamw learning_rate=1e-4 max_grad_norm=None flux_lora_target=Before it was committed lr_warmpup_steps=1
    # 16.6s/it 3k steps 13:40 hours memory

    if tune.user_id == 2:
        tune.report_to = "wandb"

    for k, v in os.environ.items():
        if k.startswith('TR_'):
            setattr(tune, k[3:].lower(), v)
            print(f"Setting {k[3:].lower()}={v}")
    if os.environ.get('TR_DISABLE_FACE_CROP'):
        tune.face_crop = False

    if tune.report_to == 'wandb' and not tune.validation_steps:
        tune.validation_steps = 100
    elif not tune.validation_steps:
        tune.validation_steps = 50000

    steps = min(5000, int(tune.steps) if tune.steps else 2000)

    print(f"output_dir={output_dir} steps={steps}")

    num_gpus = 1 # torch.cuda.device_count()
    # accelerate will accumulate across processes/gpus and so we need to divide by num_gpus
    # len(tune.orig_images) - max train_batch should not be bigger than amount of images
    if tune.train_batch:
        train_batch = int(tune.train_batch) // num_gpus
        if train_batch > len(tune.orig_images):
            train_batch = len(tune.orig_images)
    else:
        train_batch = max(1, (min(min(4, len(tune.orig_images)), 4))) // num_gpus

    if tune.preprocessing=='2':
        print("Using preprocessing v2")
        data_backend_config, resolution = create_data_config_v2(tune, output_dir)
    else:
        print("Using preprocessing v1")
        data_backend_config, resolution = create_data_config_v1(tune, output_dir)
    caption_strategy = tune.caption_strategy or "instanceprompt"

    torch.cuda.empty_cache()
    os.environ['SIMPLETUNER_CONFIG_BACKEND'] = 'cmd'
    os.environ['SIMPLETUNER_LOG_LEVEL'] = 'DEBUG'
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    os.environ['DEBUG_LOG_FILENAME'] = f"{output_dir}/debug.log"
    if tune.preprocessing=='2':
        tail_lines = run_with_output([
            'accelerate',
            'launch',
            '--gpu_ids', CUDA_VISIBLE_DEVICES,
            # '--mixed_precision=no',
            # *([f'--multi_gpu'] if num_gpus > 1 else []),
            # f'--num_processes={num_gpus}',
            '--num_machines=1',
            '--dynamo_backend=no',
            'train.py',
            # '--base_model_default_dtype=fp32',
            '--model_type=lora',
            *(['--flux_guidance_mode', tune.flux_guidance_mode] if tune.flux_guidance_mode else []),
            '--pretrained_model_name_or_path', model_path,
            *([
                '--pretrained_transformer_model_name_or_path', download_dev2pro(),
                '--pretrained_transformer_subfolder', 'none',
            ] if tune.dev2pro else []),
            '--enable_xformers_memory_efficient_attention', # ?
            '--gradient_checkpointing', # avoid OOM
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
            f'--optimizer={tune.optimizer or "adamw_bf16"}',
            f'--lora_type', tune.lora_type or 'standard',
            '--init_lokr_norm',
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
            '--user_prompt_library', create_prompt_library(tune, output_dir),
            '--model_family=flux',
            f'--train_batch={train_batch}',
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
            '--checkpointing_steps', str(tune.checkpointing_steps or 100),
            '--checkpoints_total_limit=10',
            '--validation_steps', str(tune.validation_steps) if tune.validation_steps else '5000',
            f'--tracker_run_name={tune.id}-{tune.branch}-{os.environ.get("TRACKER_NAME", timestamp)} {tune.title} {tune.args}',
            *(['--evaluation_type=face'] if tune.report_to else []),
            '--tracker_project_name=flux-lora',
            '--validation_guidance=3.5',
            '--validation_guidance_rescale=0.0',
            '--disable_benchmark',
            *(['--flux_schedule_auto_shift'] if tune.flux_schedule_auto_shift else []),
            '--flux_schedule_shift', str(tune.flux_schedule_shift if tune.flux_schedule_shift is not None else 0),
            '--skip_file_discovery=aspect,metadata',
            *(['--prepend_instance_prompt'] if caption_strategy == "textfile" else []),
            *(['--flux_attention_masked_training'] if tune.segmentation else []),
        ])
    else:
        tail_lines = run_with_output([
            'accelerate',
            'launch',
            '--mixed_precision=no',
            '--gpu_ids', CUDA_VISIBLE_DEVICES,
            *([f'--multi_gpu'] if num_gpus > 1 else []),
            f'--num_processes={num_gpus}',
            '--num_machines=1',
            '--dynamo_backend=no',
            'simpletuner_v0/train.py',
            '--base_model_default_dtype=fp32',
            '--model_type=lora',
            '--pretrained_model_name_or_path', model_path,
            '--enable_xformers_memory_efficient_attention',
            '--gradient_checkpointing',
            '--set_grads_to_none',
            '--gradient_accumulation_steps=1',
            '--resume_from_checkpoint=latest',
            '--snr_gamma=5',
            '--data_backend_config', data_backend_config,
            '--num_train_epochs=0',
            f'--max_train_steps={steps}',
            '--metadata_update_interval=65',
            # https://wandb.ai/astria/lora-training/runs/b94a195701ed0a7d7b53e6c9771c4388?nw=nwuserburgalonastria
            *([f'--max_grad_norm={tune.max_grad_norm}'] if tune.max_grad_norm else []),
            # default to adamw
            *(['--use_prodigy_optimizer'] if tune.optimizer=='prodigy' else []),
            # ["mmdit", "context", "all"]
            *([f'--flux_lora_target={tune.flux_lora_target}'] if tune.flux_lora_target else []),
            f'--learning_rate={tune.learning_rate or 1e-4}',
            '--lr_scheduler=constant_with_warmup',
            '--seed=42',
            '--lr_warmup_steps=10',
            '--output_dir', output_dir,
            '--inference_scheduler_timestep_spacing=trailing',
            '--training_scheduler_timestep_spacing=trailing',
            '--report_to', tune.report_to or 'tensorboard',
            '--allow_tf32',
            '--mixed_precision=bf16',
            *([f'--base_model_precision={os.environ.get("BASE_MODEL_PRECISION")}'] if os.environ.get("BASE_MODEL_PRECISION") else []),
            *([f'--base_model_precision={tune.base_model_precision}'] if tune.base_model_precision else []),
            # helps see that we're not destroying the priors
            # '--validation_disable_unconditional',
            '--i_know_what_i_am_doing',
            '--keep_vae_loaded',
            f'--lora_rank={tune.lora_rank or 64}',
            f'--lora_alpha={tune.lora_alpha or 64}',
            '--user_prompt_library', create_prompt_library(tune, output_dir),
            '--flux',
            f'--train_batch={train_batch}',
            '--max_workers=1',
            '--read_batch_size=1',
            '--write_batch_size=1',
            '--caption_dropout_probability=0.1',
            '--torch_num_threads=8',
            '--image_processing_batch_size=32',
            '--vae_batch_size=1',
            # '--validation_prompt="ohwx woman holding flowers, red sweater, studio photography, plain white background"',
            '--num_validation_images=1',
            '--validation_num_inference_steps=28',
            '--validation_seed=42',
            '--minimum_image_size=64',
            f'--resolution={resolution}',
            '--validation_resolution=1024',
            '--resolution_type=pixel',
            '--checkpointing_steps=100',
            '--checkpoints_total_limit=20',
            '--validation_steps', str(tune.validation_steps) if tune.validation_steps else '5000',
            f'--tracker_run_name={tune.id}-{tune.branch}-{timestamp}',
            '--tracker_project_name=flux-lora',
            '--validation_guidance=3.5',
            '--validation_guidance_rescale=0.0',
            *(['--prepend_instance_prompt'] if caption_strategy == "textfile" else []),
        ])

    if not os.path.exists(f"{output_dir}/pytorch_lora_weights.safetensors"):
        raise Exception(f"Failed to train {tune.id}: {tail_lines}")

    # upload checkpoint
    # copy file to models dir
    shutil.copyfile(
        f"{output_dir}/pytorch_lora_weights.safetensors",
        f"{MODELS_DIR}/{tune.id}.safetensors",
    )
    run([
        "aws", "s3", "cp",
        f'{output_dir}/pytorch_lora_weights.safetensors',
        f"s3://sdbooth2-production/models/{tune.id}.safetensors",
    ])
    server_tune_done(tune)

    # Cleanup ephemeral dir to avoid POD getting killed on exceeding disk space
    if not os.environ.get('DEBUG') and not os.environ.get('RETRAIN'):
        for f in glob.glob(f"{EPHEMERAL_MODELS_DIR}/{tune.id}-*"):
            shutil.rmtree(f, ignore_errors=True)

def train(tune: JsonObj):
    try:
        train_no_catch(tune)
    except Exception as e:
        if os.environ.get('DEBUG'):
            raise
        traceback.print_exc()
        print(f"Failed to train {tune.id}: {e}")
        report_tune_job_failure(tune, traceback.format_exc())

if __name__ == "__main__":
    def poll():
        i = 0
        max_sleeps = int(os.environ.get('MAX_SLEEPS', 90))
        while not is_terminated() and (i < max_sleeps or os.environ.get('DONT_STOP')):
            print(".", end="")
            processed_jobs = poll_train()
            if processed_jobs == 0:
                i = i + 1
                time.sleep(2)
            else:
                i = 0
        print(f"Exiting poll after i={i}")

    for id in sys.argv[1:]:
        if id == 'poll':
            poll()
            continue
        if id == 'stop':
            run(['runpodctl', 'remove', 'pod', os.environ['RUNPOD_POD_ID']])
            break


        tune = request_tune_job_from_server(id)
        print(f"Starting training for tune {tune.id}")
        train(tune)
