import os, sys
import time
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

# Ensure astria package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astria.taylor_cache.transformer import (
    FluxTransformer2DTaylorCachingModel,
    FluxTransformer2DDiCacheCachingModel,
)
from astria.astria_utils import device, MODELS_DIR

def run_inference(pipe, prompt="a photograph of a woman holding flowers", num_images=4, steps=28, seed=42, cache_overrides=None):
    images = []
    start = time.time()
    for i in range(num_images):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        this_prompt = f"{prompt}, variation {i+1}"
        extra_kwargs = {}
        if cache_overrides:
            extra_kwargs["joint_attention_kwargs"] = {"cache_overrides": cache_overrides}

        if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_number_of_steps"):
            pipe.transformer.set_number_of_steps(steps)

        image = pipe(
            prompt=this_prompt,
            num_inference_steps=steps,
            generator=generator,
            **extra_kwargs,
        ).images[0]
        images.append(image)
    elapsed = time.time() - start
    return images, elapsed

def assemble_grid(list_of_lists, nrow=4):
    def to_tensor(img):
        # Convert PIL.Image or np.ndarray to proper tensor
        if isinstance(img, np.ndarray):
            arr = img
        else:
            arr = np.array(img)

        # Handle object arrays
        if arr.dtype == np.object_:
            arr = np.array([np.array(x) for x in arr], dtype=np.uint8)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return torch.from_numpy(arr).permute(2, 0, 1)

    imgs = [to_tensor(img) for sub in list_of_lists for img in sub]
    grid = make_grid(imgs, nrow=nrow)
    grid_img = Image.fromarray(grid.permute(1, 2, 0).byte().cpu().numpy())
    return grid_img

def main():
    model_path = f"{MODELS_DIR}/1504944-flux1"
    print(f"Loading baseline pipeline from {model_path}")
    baseline_pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    
    print("Running baseline (no Taylor caching)")
    baseline_cache_file = "baseline_results.npz"
    if os.path.exists(baseline_cache_file):
        print(f"Loading cached baseline results from {baseline_cache_file}")
        data = np.load(baseline_cache_file, allow_pickle=True)
        baseline_images = list(data["images"])
        baseline_time = float(data["time"])
    else:
        baseline_images, baseline_time = run_inference(baseline_pipe)
        np.savez_compressed(
            baseline_cache_file,
            images=np.array(baseline_images, dtype=object),
            time=baseline_time
        )

    print("Loading pipeline with Taylor caching transformer")
    cache_pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    if isinstance(cache_pipe.transformer, FluxTransformer2DModel):
        # cache_pipe.transformer = FluxTransformer2DTaylorCachingModel(cache_pipe.transformer)
        cache_pipe.transformer = FluxTransformer2DDiCacheCachingModel(
            cache_pipe.transformer,
            error_choice="cosine_l1_hybrid",
            probe_depth=1,
            rel_thresh_map=[
                {"start": 0.00, "threshold": 0.05},
                {"start": 0.35, "threshold": 0.08},
                {"start": 0.45, "threshold": 0.15},
                {"start": 0.75, "threshold": 0.25},
            ],
            ret_ratio=0.0,
            max_consec_skips=8,
        )
        cache_pipe.transformer.__class__.__name__ = "FluxTransformer2DModel"

    print("Running with Taylor caching")
    # import debugpy
    # debugpy.listen(('0.0.0.0', 11566))
    # debugpy.wait_for_client()
    cache_images, cache_time = run_inference(
        cache_pipe,
        # cache_overrides={"mode": "Delta", "fresh_threshold": 4, "max_order": 2}
        # cache_overrides={
        #     "mode": "DiCache",
        #     "probe_depth": 2,
        #     "error_choice": "delta_y",
        #     "rel_l1_thresh": 0.4,
        #     "ret_ratio": 0.2
        # }
    )

    print(f"Baseline time: {baseline_time:.2f}s for 4 images ({baseline_time/3.0:.2f}s/img)")
    print(f"Taylor cache time: {cache_time:.2f}s for 4 images ({cache_time/3.0:.2f}s/img)")

    grid = assemble_grid([baseline_images, cache_images], nrow=4)
    out_path = "taylor_cache_benchmark_grid.jpg"
    grid.save(out_path)
    print(f"Saved comparison grid to {out_path}")

if __name__ == "__main__":
    main()