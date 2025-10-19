# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
SeedVR2 7B – Image Upscaling API

This module provides a small, importable API to do image upscaling using the
SeedVR2 video diffusion model (works well on images, too).

Key design points:
- Model loading and inference are separate steps (so tests can import the class, load once, and run many inferences).
- Robust imports even when the original repo was copied and package roots are not installed.
- Minimal, dependency-light surface; you can pass precomputed text embeddings (recommended) or file paths to them.

Quick usage:

    from astria.seedvr2.projects.inference_image_upscale_seedvr2_7b import SeedVR2ImageUpscaler
    import torch

    upscaler = SeedVR2ImageUpscaler(
        config_path="./configs_7b/main.yaml",
        checkpoint_path="./ckpts/seedvr2_ema_7b.pth",
        device="cuda",
        sp_size=1,            # must be 1 for images
        res_h=720, res_w=1280 # target scale "hint" (model trained for high-res)
    )
    upscaler.load()  # load models

    # Provide text embeddings (recommended: same ones used in the original inference script)
    pos = torch.load("pos_emb.pt")
    neg = torch.load("neg_emb.pt")
    upscaler.set_text_embeddings(pos, neg)

    # Run on a single image path / PIL / np.ndarray / torch.Tensor
    out = upscaler.upscale("path/to/your_image.png", seed=123, sample_steps=1, cfg_scale=1.0)

    # Save if desired
    upscaler.save_image(out, "upscaled.png")  # uint8 HxWxC

Notes:
- If you do not call `set_text_embeddings`, you can pass `pos_emb_path`/`neg_emb_path`
  directly to `upscale(...)`. Text-string prompting is not implemented here because
  the repo’s text encoder wiring is not exposed in this minimal API.
"""

from __future__ import annotations

import os
import sys
import gc
import types
import warnings

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import numpy as np

from PIL import Image

from astria.astria_utils import CACHE_DIR
from astria.seedvr2.download import ensure_seedvr2_7b_checkpoint, ensure_seedvr2_vae_checkpoint


# Optional for saving – same dep used in the original scripts
try:
    import mediapy  # type: ignore
except Exception:
    mediapy = None  # saving still possible via numpy return


from einops import rearrange
from omegaconf import OmegaConf

# Registers DIT architecture pieces (side-effect import in original code)
from astria.seedvr2.models.dit import na  # noqa: F401

from astria.seedvr2.data.image.transforms.divisible_crop import DivisibleCrop
from astria.seedvr2.data.image.transforms.na_resize import NaResize
from astria.seedvr2.data.video.transforms.rearrange import Rearrange as VidRearrange

from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io import read_image

from astria.seedvr2.common.distributed import get_device, init_torch
from astria.seedvr2.common.seed import set_seed
from astria.seedvr2.common.config import load_config
from astria.seedvr2.common.distributed.ops import sync_data

from astria.seedvr2.projects.video_diffusion_sr.infer import VideoDiffusionInfer

from astria.seedvr2.common.colorfix import wavelet_reconstruction


# ------------------------------ Utilities ---------------------------------------

# add near the top

def _resolve_config_path(explicit: Optional[Union[str, Path]] = None) -> Path:
    """
    Find configs_7b/main.yaml robustly.

    Priority:
      1) explicit path (file or directory)
      2) env SEEDVR2_CONFIG_PATH (file or directory)
      3) alongside this module: astria/seedvr2/configs_7b/main.yaml
      4) CWD: ./configs_7b/main.yaml
      5) importlib.resources (when packaged)

    Returns a Path to the YAML file or raises FileNotFoundError.
    """
    def _normalize(p: Union[str, Path]) -> Path:
        p = Path(p)
        if p.is_dir():
            p = p / "main.yaml"
        return p

    # 1) explicit
    if explicit is not None:
        p = _normalize(explicit)
        if p.is_file():
            return p.resolve()

    # 2) env
    env_val = os.getenv("SEEDVR2_CONFIG_PATH")
    if env_val:
        p = _normalize(env_val)
        if p.is_file():
            return p.resolve()

    # 3) alongside this module
    here = Path(__file__).resolve().parent
    candidate = here / "configs_7b" / "main.yaml"
    if candidate.is_file():
        return candidate

    # 4) CWD
    candidate = Path.cwd() / "configs_7b" / "main.yaml"
    if candidate.is_file():
        return candidate.resolve()

    # 5) packaged resource (optional)
    try:
        from importlib.resources import files as ir_files  # py>=3.9
        pkg_candidate = ir_files("astria.seedvr2") / "configs_7b" / "main.yaml"
        # pkg_candidate is a Traversable; turn into real path if possible:
        candidate = Path(str(pkg_candidate))
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate SeedVR2 config 'configs_7b/main.yaml'. "
        "Tried explicit path, SEEDVR2_CONFIG_PATH, alongside module, CWD, and package resources."
    )


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _to_tchw(
    img: Union[str, "np.ndarray", "torch.Tensor", "Image.Image"]
) -> torch.Tensor:
    """
    Normalize various image inputs to torch float32 in [0,1] with shape (T,C,H,W), T=1.
    Accepts: filepath (str), numpy array (H,W,C or C,H,W; uint8/float), torch tensor (same).
    """
    if isinstance(img, str):
        if not _is_image_file(img):
            raise ValueError(f"Unsupported path or extension for image: {img}")
        t = read_image(img)  # (C,H,W) uint8
        t = t.unsqueeze(0).float() / 255.0  # (1,C,H,W) float32 [0,1]
        return t

    if isinstance(img, Image.Image):
        # Convert any PIL mode to 3-channel RGB (most stable for the pipeline)
        img = img.convert("RGB")
        img = np.array(img)  # HWC uint8
        # fall through to NumPy handling below

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:  # H,W (grayscale) -> H,W,1
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"NumPy image must be HxWxC or CxHxW; got shape {img.shape}")
        # If channel-first, convert to HWC
        if arr.shape[0] in {1, 3} and arr.shape[2] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.clip(arr, 0.0, 1.0)
        # HWC -> (1,C,H,W)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return torch.from_numpy(arr)

    if torch.is_tensor(img):
        t = img
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)  # 1x1xH xW
        elif t.ndim == 3:
            # Assume CHW or HWC -> normalize to CHW
            if t.shape[0] not in {1, 3} and t.shape[2] in {1, 3}:
                t = t.permute(2, 0, 1)  # HWC -> CHW
            t = t.unsqueeze(0)  # (1,C,H,W)
        elif t.ndim == 4:
            # If already TCHW or BCHW; we only accept batch=1 or T=1
            if t.shape[0] != 1:
                raise ValueError(f"Expected a single image (T=1), got tensor shape {tuple(t.shape)}")
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}")

        t = t.float()
        if t.max() > 1.0 or t.min() < 0.0:
            # assume 0..255 if integer-like values present
            if t.dtype.is_floating_point:
                t = torch.clamp(t, 0, 255)
            t = t / 255.0
        return t

    raise TypeError(f"Unsupported image input type: {type(img)}")


# ------------------------------ Main API ----------------------------------------

class SeedVR2ImageUpscaler:
    """
    Image upscaling helper around SeedVR2 7B.

    Methods:
        load():             load/initialize config + models (separate from inference)
        set_text_embeddings(pos, neg) / clear_text_embeddings()
        upscale(image, ...): run single-image enhancement/upscaling and return uint8 HxWxC ndarray
        save_image(image, path): convenience writer (uses mediapy if available)

    Notes:
        - sp_size MUST be 1 for image inputs.
        - You must provide text embeddings (pos/neg) either via `set_text_embeddings`
          or via `pos_emb_path`/`neg_emb_path` arguments to `upscale(...)`.
    """
    from astria.seedvr2.download import DEFAULT_FILENAME_SHARP

    def __init__(
        self,
        config_path: str|None = None,
        checkpoint_path: str = f"{CACHE_DIR}/{DEFAULT_FILENAME_SHARP}",
        device: Optional[str] = 'cuda',
        sp_size: int = 1,
        res_h: int = 2160,
        res_w: int = 3840,
    ) -> None:
        self.config_path = _resolve_config_path(config_path)
        self.checkpoint_path = checkpoint_path
        self.sp_size = int(sp_size)
        if self.sp_size != 1:
            raise ValueError("`sp_size` must be 1 for image inputs.")
        self.res_h = int(res_h)
        self.res_w = int(res_w)

        self._device = device or "cuda"
        self.runner: Optional[VideoDiffusionInfer] = None
        self._text_pos: Optional[torch.Tensor] = None
        self._text_neg: Optional[torch.Tensor] = None

        # Build transforms once
        # Pipeline follows the original: TCHW -> clamp -> crop-divisible-by-16 -> normalize -> to CTHW
        self._video_transform = Compose(
            [
                NaResize(
                    resolution=(self.res_h * self.res_w) ** 0.5,
                    mode="area",
                    downsample_only=False,  # model trained for high-res; allow upsampling
                ),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                VidRearrange("t c h w -> c t h w"),
            ]
        )

    # --------- Model lifecycle ---------

    def load(self) -> None:
        """
        Load config + models. Safe to call once and reuse for many inferences.
        """
        if self.runner is not None:
            return

        # Init torch runtime; long timeout as in original
        import datetime
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))

        # Config + runner
        config = load_config(str(self.config_path))
        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)

        here = Path(__file__).resolve().parent  # astria/seedvr2
        vae_yaml = here / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
        if vae_yaml.is_file():
            runner.config.vae.model.__inherit__ = str(vae_yaml)

        # 2) Ensure/download VAE weight into CACHE_DIR and patch the config
        vae_ckpt_path = ensure_seedvr2_vae_checkpoint(target_dir=str(CACHE_DIR))
        runner.config.vae.checkpoint = str(vae_ckpt_path)

        # Models
        ckpt_path, pos_path, neg_path = ensure_seedvr2_7b_checkpoint(self.checkpoint_path)
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.checkpoint_path = ckpt_path  # keep the resolved path
        runner.configure_dit_model(device=self._device, checkpoint=ckpt_path)
        runner.configure_vae_model()

        # VAE memory limit if exposed
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        self.runner = runner

    def set_text_embeddings(self, pos: torch.Tensor, neg: torch.Tensor) -> None:
        """
        Store precomputed text embeddings. Expected tensors should match the repo's
        `pos_emb.pt` / `neg_emb.pt` shapes for SeedVR2.
        """
        self._text_pos = pos
        self._text_neg = neg

    def clear_text_embeddings(self) -> None:
        self._text_pos = None
        self._text_neg = None

    # --------- Inference ---------

    @torch.inference_mode()
    def upscale(
        self,
        image: Union[str, "np.ndarray", "torch.Tensor", "Image.Image"],
        *,
        seed: int = 666,
        sample_steps: int = 1,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        pos_emb_path: Optional[str] = None,
        neg_emb_path: Optional[str] = None,
        return_torch: bool = False,
        return_pil: bool = True,
        model_offloading: bool = False,
        dit_offload: bool = False,
    ) -> Union["np.ndarray", "torch.Tensor", "Image.Image"]:
        """
        Run single-image enhancement/upscaling and return an uint8 HxWxC numpy array (default).

        Args:
            image: str path, numpy array (HWC/CHW), or torch tensor (CHW/HWC). Will be normalized to TCHW (T=1).
            seed: RNG seed (shared across ranks, parity with original script).
            sample_steps: diffusion sampling steps (1 by default, mirroring the fast path).
            cfg_scale: classifier-free guidance scale.
            cfg_rescale: guidance rescale.
            pos_emb_path / neg_emb_path: If text embeddings weren’t set beforehand, load them from files here.
            return_torch: If True, returns a torch.uint8 tensor (H,W,C) on CPU; else returns numpy uint8.

        Returns:
            numpy.ndarray (H, W, C), dtype=uint8 by default; or torch.Tensor if return_torch=True.
        """
        if self.runner is None:
            raise RuntimeError("Model not loaded. Call `load()` before `upscale()`.")

        runner = self.runner
        if pos_emb_path is None:
            pos_emb_path = self.pos_path
        if neg_emb_path is None:
            neg_emb_path = self.neg_path

        # Seed, CFG, sampler
        set_seed(seed, same_across_ranks=True)
        runner.config.diffusion.cfg.scale = float(cfg_scale)
        runner.config.diffusion.cfg.rescale = float(cfg_rescale)
        runner.config.diffusion.timesteps.sampling.steps = int(sample_steps)
        runner.configure_diffusion()

        # Prepare text embeddings
        texts_pos, texts_neg = self._resolve_text_embeds(pos_emb_path, neg_emb_path)
        # ensure list wrapper as expected by runner.inference signature
        text_embeds_dict = {"texts_pos": [texts_pos], "texts_neg": [texts_neg]}

        # Input -> TCHW float [0,1]
        tchw = _to_tchw(image)
        ori_h = tchw.shape[-2]
        ori_w = tchw.shape[-1]

        # Transform -> CTHW normalized [-1,1]
        cond = self._video_transform(tchw.to(self._device))  # C T H W
        ori_length = cond.shape[1]  # T
        if ori_length != 1:
            raise ValueError(f"Expected single-frame image, got T={ori_length}")

        # Move models to appropriate devices for encode/infer, mirroring original memory pattern
        if model_offloading:
            runner.dit.to("cpu")
            runner.vae.to(self._device)
        cond_latents = runner.vae_encode([cond])  # list of latents
        if model_offloading:
            runner.vae.to("cpu")
            runner.dit.to(self._device)

        # Prepare noises/conditions and run the diffusion model
        samples = self._generation_step(runner, text_embeds_dict, cond_latents, dit_offload=dit_offload)

        # samples: list with a single item – shape (T,C,H,W) with values in [-1,1]
        sample = samples[0]
        if sample.shape[0] > 1:
            sample = sample[:1]  # clamp to one frame

        # Optional color fix vs. direct
        # Build input frames in T,C,H,W for color fix
        inp = rearrange(cond, "c t h w -> t c h w")
        try:
            sample = wavelet_reconstruction(sample.to("cpu"), inp[: sample.size(0)].to("cpu"))
        except Exception:
            warnings.warn("Color fix failed; returning raw output.")
        sample = sample.to("cpu")

        # Convert to uint8 HxWxC
        # (either 3D TCHW or 4D TCHW where T=1)
        if sample.ndim == 3:  # C,H,W (no T)
            chw = sample
        else:  # T,C,H,W with T=1
            chw = sample[0]
        hwc = rearrange(chw, "c h w -> h w c")

        # Scale from [-1,1] -> [0,255]
        hwc = hwc.clamp_(-1, 1).mul_(0.5).add_(0.5).mul_(255).round_().to(torch.uint8)

        if return_pil and return_torch:
            raise ValueError("Choose only one of return_pil or return_torch.")

        if return_pil:
            return self.as_pil(hwc)

        # Return
        if return_torch:
            return hwc  # CPU uint8 tensor HxWxC
        return hwc.numpy()

    # --------- Helpers ---------

    def _resolve_text_embeds(
        self,
        pos_emb_path: Optional[str],
        neg_emb_path: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Choose embeddings in priority order:
            1) Already set via set_text_embeddings
            2) Load from provided file paths
            3) Load from default 'pos_emb.pt'/'neg_emb.pt' if present
        """
        if self._text_pos is not None and self._text_neg is not None:
            return self._text_pos.to(self._device), self._text_neg.to(self._device)

        # If explicit paths provided, load them
        if pos_emb_path is not None and neg_emb_path is not None:
            pos = torch.load(pos_emb_path, map_location=self._device)
            neg = torch.load(neg_emb_path, map_location=self._device)
            return pos, neg

        # Try local defaults for parity with original script
        cwd_pos, cwd_neg = "pos_emb.pt", "neg_emb.pt"
        if os.path.exists(cwd_pos) and os.path.exists(cwd_neg):
            pos = torch.load(cwd_pos, map_location=self._device)
            neg = torch.load(cwd_neg, map_location=self._device)
            return pos, neg

        raise RuntimeError(
            "Text embeddings not provided. Call `set_text_embeddings(pos, neg)` or pass "
            "`pos_emb_path` and `neg_emb_path` to `upscale(...)`."
        )

    def _generation_step(
        self,
        runner: VideoDiffusionInfer,
        text_embeds_dict: Dict[str, Iterable[torch.Tensor]],
        cond_latents: Iterable[torch.Tensor],
        dit_offload: bool = False,
    ) -> Iterable[torch.Tensor]:
        """
        Mirrors the original `generation_step` with single-item semantics for images.
        """
        def _move_to_cuda(x):
            return [i.to(get_device()) for i in x]

        cond_latents = list(cond_latents)
        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]

        noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
        noises, aug_noises, cond_latents = list(map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents)))

        cond_noise_scale = 0.0

        def _add_noise(x, aug_noise):
            t = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
            shape = torch.tensor(x.shape[1:], device=get_device())[None]
            t = runner.timestep_transform(t, shape)
            x = runner.schedule.forward(x, aug_noise, t)
            return x

        conditions = [
            runner.get_condition(
                noise,
                task="sr",
                latent_blur=_add_noise(latent_blur, aug_noise),
            )
            for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
        ]

        # Ensure embeds are on device
        for i, emb in enumerate(text_embeds_dict["texts_pos"]):
            text_embeds_dict["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds_dict["texts_neg"]):
            text_embeds_dict["texts_neg"][i] = emb.to(get_device())

        with torch.no_grad(), torch.autocast(self._device, torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=dit_offload,
                **text_embeds_dict,
            )

        samples = [
            (
                rearrange(video[:, None], "c t h w -> t c h w")
                if video.ndim == 3
                else rearrange(video, "c t h w -> t c h w")
            )
            for video in video_tensors
        ]
        del video_tensors
        gc.collect()
        torch.cuda.empty_cache()
        return samples

    # --------- I/O convenience ---------

    @staticmethod
    def save_image(img: Union["np.ndarray", "torch.Tensor"], path: str) -> None:
        """
        Save uint8 HxWxC to disk. Uses mediapy if available; otherwise falls back
        to a tiny Pillow-free writer via imageio if installed, else raises.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.dtype != np.uint8:
            raise ValueError("save_image expects uint8 image array.")
        if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
            raise ValueError(f"save_image expects HxWxC, C in [1,3,4]; got {img.shape}")

        if mediapy is not None:
            mediapy.write_image(path, img)
            return

        try:
            import imageio  # type: ignore
            imageio.imwrite(path, img)
            return
        except Exception as e:
            raise RuntimeError(
                "Unable to save image: mediapy not installed and imageio fallback failed."
            ) from e

    @staticmethod
    def as_pil(
        img: "np.ndarray | torch.Tensor",
        *,
        assume_float01: bool = True,
    ) -> "Image.Image":
        """
        Convert an image (uint8 HxWxC or float HxWxC) to a PIL.Image without saving.

        - Accepts numpy or torch tensors (HWC). If float, assumes [0,1] by default.
        - Supports 1, 3, or 4 channels -> L / RGB / RGBA respectively.
        """
        if Image is None:
            raise ImportError("Pillow is not installed. `pip install pillow`")

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        if img.ndim == 2:  # grayscale HxW
            mode = "L"
            arr = img
            if arr.dtype != np.uint8:
                if not np.issubdtype(arr.dtype, np.floating):
                    raise ValueError("Expected float or uint8 for grayscale array.")
                arr = np.clip(arr, 0.0, 1.0) if assume_float01 else np.clip(arr, 0, 255)
                arr = (arr * 255.0).round().astype(np.uint8) if assume_float01 else arr.astype(np.uint8)
            return Image.fromarray(arr, mode=mode)

        if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected HxWxC with C in (1,3,4); got shape {img.shape}")

        c = img.shape[2]
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[c]

        arr = img
        if arr.dtype != np.uint8:
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError("Only float or uint8 arrays are supported.")
            arr = np.clip(arr, 0.0, 1.0) if assume_float01 else np.clip(arr, 0, 255)
            arr = (arr * 255.0).round().astype(np.uint8) if assume_float01 else arr.astype(np.uint8)

        if c == 1:
            arr = arr[..., 0]  # HxW for mode 'L'
        return Image.fromarray(arr, mode=mode)