import os, sys

sys.path.append("astria")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
from pathlib import Path
from astria_tests.test_infer import IMG_POSE, BASE_PROMPT, FLUX_LORA, run_images, FLUX_FACEID, pipe

# The upscaler class you placed here:
from seedvr2.upscaler import SeedVR2ImageUpscaler
from astria_utils import JsonObj, MODELS_DIR
from image_utils import load_image



def test_seedvr2_image_upscale():
    up = SeedVR2ImageUpscaler()
    up.load()
    # img = load_image('https://mp.astria.ai/ry34z8d9nnmti9eizrs066ty2yl4')
    out = up.upscale(IMG_POSE, seed=0, sample_steps=1, cfg_scale=1.0)
    out.save(MODELS_DIR + f"/test_seedvr2_image_upscale.png")

# def test_seedvr2_image_upscale_profile():
#     import cProfile, pstats, io, torch, re
#     from astria.seedvr2.upscaler import SeedVR2ImageUpscaler
#     from astria.astria_utils import MODELS_DIR

#     up = SeedVR2ImageUpscaler()
#     up.load()

#     # Warmup so caches/JIT don’t skew
#     _ = up.upscale(IMG_POSE, seed=0, sample_steps=1, cfg_scale=1.0)
#     torch.cuda.synchronize()

#     pr = cProfile.Profile()
#     pr.enable()

#     out = up.upscale(IMG_POSE, seed=0, sample_steps=1, cfg_scale=1.0)
#     torch.cuda.synchronize()           # <-- crucial: include GPU time
#     pr.disable()

#     s = io.StringIO()
#     ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')  # no strip_dirs -> regex can match paths

#     # 1) Top 50 overall
#     ps.print_stats(50)

#     # 2) Narrow to the files/functions you care about
#     #    Since we didn't strip dirs, these will match full paths like '.../astria/seedvr2/...'
#     ps.print_stats(r'astria/seedvr2|tensor_bundles|video_diffusion_sr|upscaler\.py', 60)

#     # 3) Who calls your hot funcs / what they call
#     ps.print_callers(r"\{method 'to' of 'torch\._C\.TensorBase' objects\}")
#     ps.print_callers(r"torch\.nn\.modules\.module\.Module\.to")

#     report_path = f"{MODELS_DIR}/seedvr2_cprofile.txt"
#     with open(report_path, "w") as f:
#         f.write(s.getvalue())
#     print(s.getvalue())
#     print(f"[cProfile] Wrote detailed report to {report_path}")

#     out.save(MODELS_DIR + "/test_seedvr2_image_upscale.png")

def test_seedvr2_pipe():
    pipe.reset(True)
    input_images = [load_image(t) for t in [
        'https://mp.astria.ai/ry34z8d9nnmti9eizrs066ty2yl4',
        # 'https://mp.astria.ai/z9bir6muno34znw1ar8f29p0ocul',
        # 'https://mp.astria.ai/80alvnq3rqijf7n1f5lz5xxum37v',
        # 'https://mp.astria.ai/p0samx6m8hnslit3l6spnzqgqlpt',
    ]]
    images = pipe.upscale_seedvr2(input_images, BASE_PROMPT)
    # for 1 image 14MP
    # T#1504944 P#test-prompt-id upscale seedvr2 1 14.30 seconds
    for i, image in enumerate(images):
        image.save(MODELS_DIR + f"/test_seedvr2_pipe-{i}.png")

def test_txt2img_lora_upscale_seedvr2():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        inpaint_faces=True,
        upscale_v4=True,
        w=1072,
        h=1344,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-before-upscale')

    # prompt = JsonObj(
    #     **copy.copy(BASE_PROMPT.__dict__),
    #     w=896,
    #     h=1152,
    #     super_resolution=True,
    # )
    # prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    # prompt.tunes=[FLUX_LORA]
    # # import debugpy
    # # debugpy.listen(('0.0.0.0', 11566))
    # # debugpy.wait_for_client()
    # run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-upscaled')
    #
    #
    # # Check post super-resolution everything is okay
    # prompt = JsonObj(
    #     **copy.copy(BASE_PROMPT.__dict__),
    #     super_resolution=True,
    #     inpaint_faces=True,
    # )
    #
    # prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    # prompt.tunes=[FLUX_LORA]
    # run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-after-upscale')


def test_upscale_stage_1():
    print("test_upscale_stage_1")
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        upscale_v4=True,
        images=[IMG_POSE],
        stage=1,
    )
    run_images(prompt)

def test_txt2img_lora_upscale_seedvr2():
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        inpaint_faces=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-before-upscale')

    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        w=896,
        h=1152,
        super_resolution=True,
        upscale_v4=True,
    )
    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-upscaled')


    # Check post super-resolution everything is okay
    prompt = JsonObj(
        **copy.copy(BASE_PROMPT.__dict__),
        super_resolution=True,
        inpaint_faces=True,
        upscale_v4=True,
    )

    prompt.text=f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} woman holding flowers --upscale_v4"
    prompt.tunes=[FLUX_LORA]
    run_images(prompt, 'test_txt2img_lora_upscale_seedvr2-after-upscale')

def test_seedvr2_batch_inpaint_upscale_v4():
    # 4 images, face inpaint + seedvr4 upscale
    prompt = JsonObj(
        id="test-prompt-id",
        tune_id=1504944,
        tunes=[],
        num_images=4,
        super_resolution=True,
        inpaint_faces=True,
        upscale_v4=True,
        w=1152,
        h=1536,
    )
    prompt.text = f"<lora:{FLUX_LORA.id}:1> {FLUX_LORA.train_token} portrait photo, looking at camera"
    prompt.tunes = [FLUX_LORA]

    # Run the usual pipeline helper (this also saves to MODELS_DIR),
    # but we also save explicit JPEGs into the test folder as requested.
    images = run_images(prompt, "seedvr2_batch_inpaint_upscale_v4")
