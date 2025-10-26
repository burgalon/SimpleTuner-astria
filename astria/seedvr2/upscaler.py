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
from astria.tensorize import seedvr2_load_or_tensorize
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


def _gpu_hist_match_uint8(src_hwc_u8: torch.Tensor,
                          ref_hwc_u8: torch.Tensor,
                          downsample: int = 4) -> torch.Tensor:
    """
    Histogram-match src to ref on GPU. Both are uint8 HxWxC (C=3), CUDA tensors.
    We compute LUTs per channel (0..255) using downsampled histograms,
    then apply LUTs to the full-resolution src on GPU.

    Returns: uint8 HxWxC (CUDA)
    """
    assert src_hwc_u8.dtype == torch.uint8 and ref_hwc_u8.dtype == torch.uint8
    assert src_hwc_u8.device.type == "cuda" and ref_hwc_u8.device.type == "cuda"
    assert src_hwc_u8.shape[-1] == ref_hwc_u8.shape[-1] == 3, "expects RGB"

    if downsample > 1:
        src_ds = src_hwc_u8[::downsample, ::downsample]
        ref_ds = ref_hwc_u8[::downsample, ::downsample]
    else:
        src_ds, ref_ds = src_hwc_u8, ref_hwc_u8

    luts = []
    for c in range(3):
        s = src_ds[..., c].reshape(-1).to(torch.int64)
        r = ref_ds[..., c].reshape(-1).to(torch.int64)
        hist_s = torch.bincount(s, minlength=256)
        hist_r = torch.bincount(r, minlength=256)
        cdf_s = hist_s.cumsum(0).float()
        cdf_r = hist_r.cumsum(0).float()
        cdf_s /= cdf_s[-1].clamp_min(1)
        cdf_r /= cdf_r[-1].clamp_min(1)
        # LUT[v] = smallest j with CDF_ref[j] >= CDF_src[v]
        lut_c = torch.searchsorted(cdf_r, cdf_s).clamp_(0, 255).to(torch.uint8)
        luts.append(lut_c)
    lut = torch.stack(luts, 0)  # (3,256)

    H, W, _ = src_hwc_u8.shape
    src_flat = src_hwc_u8.reshape(-1, 3).long().transpose(0, 1)      # (3, N)
    matched_flat = torch.gather(lut, 1, src_flat)                     # (3, N)
    matched = matched_flat.transpose(0, 1).reshape(H, W, 3).to(torch.uint8)
    return matched



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
        checkpoint_path: str = CACHE_DIR,
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
    def load(
        self,
        vae_decode_conv_max_mem: float = 8.0,
        vae_decode_norm_max_mem: float = 4.0,
    ) -> None:
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

        # # 2) Ensure/download VAE weight into CACHE_DIR and patch the config
        vae_ckpt_path = ensure_seedvr2_vae_checkpoint(target_dir=str(CACHE_DIR))
        # runner.config.vae.checkpoint = str(vae_ckpt_path)

        # # Models
        ckpt_path, pos_path, neg_path = ensure_seedvr2_7b_checkpoint(self.checkpoint_path)
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.checkpoint_path = ckpt_path  # keep the resolved path

        # LEGACY (SLOW) LOADING
        # runner.configure_dit_model(device=self._device, checkpoint=ckpt_path)
        # runner.configure_vae_model()

        # # VAE memory limit if exposed
        # if hasattr(runner.vae, "set_memory_limit"):
        #     runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        # self.runner = runner

        # --- FAST PATH: tensorizer bundle ---
        import time
        start = time.time()
        dit, vae = seedvr2_load_or_tensorize(
            config_yaml=str(self.config_path),
            dit_ckpt=str(ckpt_path),
            vae_ckpt=str(vae_ckpt_path),
            device=self._device,
            force_rebuild=False,
        )
        print(f'inited seedvr2 in {time.time() - start} seconds')
        runner.dit = dit

        vae.requires_grad_(False).eval()
        # optional: respect any memory/slicing knobs from the config
        if hasattr(runner.config.vae, "slicing") and hasattr(vae, "set_causal_slicing"):
            vae.set_causal_slicing(**runner.config.vae.slicing)
        # if hasattr(vae, "set_memory_limit") and hasattr(runner.config.vae, "memory_limit"):
        #     vae.set_memory_limit(**runner.config.vae.memory_limit)
        vae.set_memory_limit(
            conv_max_mem=vae_decode_conv_max_mem,
            norm_max_mem=vae_decode_norm_max_mem,
        )
        runner.vae = vae

        from astria.seedvr2.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
        from astria.seedvr2.models.video_vae_v3.modules.inflated_layers import InflatedCausalConv3d as InflatedCausalConv3dV2
        for m in vae.modules():
            if isinstance(m, InflatedCausalConv3d) or isinstance(m, InflatedCausalConv3dV2):
                m.set_memory_device("same")

        runner.config.dit.gradient_checkpoint = False
        dit.set_gradient_checkpointing(False)

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
        slow_color_fix: bool = False,
    ) -> Union["np.ndarray", "torch.Tensor", "Image.Image"]:
        """
        Run single-image enhancement/upscaling and return an uint8 HxWxC image.

        Args:
            image: str path, numpy array (HWC/CHW), or torch tensor (CHW/HWC). Will be normalized to TCHW (T=1).
            seed: RNG seed (shared across ranks).
            sample_steps: diffusion sampling steps (default 1).
            cfg_scale / cfg_rescale: classifier-free guidance knobs.
            pos_emb_path / neg_emb_path: paths to text embeddings if not set via set_text_embeddings().
            return_torch: If True, returns a torch.uint8 tensor (H,W,C) on CPU.
            return_pil: If True, returns a PIL.Image.
            model_offloading / dit_offload: optional memory knobs.
            slow_color_fix: if True, use wavelet-based color fix (slow); otherwise fast histogram match.
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

        # Prepare text embeddings (as tensors)
        texts_pos, texts_neg = self._resolve_text_embeds(pos_emb_path, neg_emb_path)
        text_embeds_dict = {"texts_pos": [texts_pos], "texts_neg": [texts_neg]}

        # Input -> TCHW float [0,1] (CPU)
        tchw = _to_tchw(image)  # (1,C,H,W)

        # Keep original input as uint8 HWC on CPU for fast color matching (reference)
        orig_uint8 = (
            tchw[0].permute(1, 2, 0)      # CHW -> HWC
            .mul(255).round().clamp(0, 255)
            .to(torch.uint8).cpu().numpy()
        )

        # Transform -> CTHW normalized [-1,1] (GPU)
        cond = self._video_transform(tchw.to(self._device))  # C T H W
        if cond.shape[1] != 1:
            raise ValueError(f"Expected single-frame image, got T={cond.shape[1]}")

        # Encode with VAE (optionally offloading)
        if model_offloading:
            runner.dit.to("cpu")
            runner.vae.to(self._device)
        cond_latents = runner.vae_encode([cond])  # list of latents
        if model_offloading:
            runner.vae.to("cpu")
            runner.dit.to(self._device)

        # Diffusion step (DiT)
        samples = self._generation_step(runner, text_embeds_dict, cond_latents, dit_offload=dit_offload)
        sample = samples[0]  # T,C,H,W in [-1,1]
        if sample.shape[0] > 1:
            sample = sample[:1]  # T=1

        # ---- Color fix ----
        if slow_color_fix:
            # Original wavelet path (CPU) — slower but potentially higher fidelity
            inp = rearrange(cond, "c t h w -> t c h w")
            try:
                sample = wavelet_reconstruction(sample.to("cpu"), inp[: sample.size(0)].to("cpu"))
            except Exception:
                warnings.warn("Color fix failed; returning raw output.")

            # Convert to uint8 HWC on CPU
            chw = sample if sample.ndim == 3 else sample[0]  # C,H,W
            hwc = rearrange(chw, "c h w -> h w c")
            final_uint8 = (
                hwc.clamp_(-1, 1).mul_(0.5).add_(0.5).mul_(255).round_()
                .to(torch.uint8).cpu().numpy()
            )
        else:
            ref_hwc_u8_gpu = (
                rearrange(tchw.to(self._device)[0], "c h w -> h w c")
                .clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            )

            # Model output -> uint8 on GPU
            out_chw = sample if sample.ndim == 3 else sample[0]  # C,H,W
            out_hwc_u8_gpu = (
                rearrange(out_chw, "c h w -> h w c")
                .clamp_(-1, 1).mul_(0.5).add_(0.5).mul_(255).round_()
                .to(torch.uint8)
            )

            # Histogram match entirely on GPU; one LUT per channel
            matched_u8_gpu = _gpu_hist_match_uint8(out_hwc_u8_gpu, ref_hwc_u8_gpu, downsample=4)

            # Single final copy for return
            final_uint8 = matched_u8_gpu.cpu().numpy()

        # ---- Return in requested format ----
        if return_pil and return_torch:
            raise ValueError("Choose only one of return_pil or return_torch.")

        if return_pil:
            return self.as_pil(final_uint8)

        if return_torch:
            return torch.from_numpy(final_uint8)  # CPU uint8 HxWxC

        return final_uint8  # numpy uint8 HxWxC

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
        *,
        cond_noise_scale: float = 0.0,      # knob (0 = skip)
        aggressive_gc: bool = False,        # keep False for speed
    ) -> Iterable[torch.Tensor]:
        """
        Mirrors the original generation_step but:
        - short-circuits blur/noise when cond_noise_scale == 0
        - avoids extra device moves / lambda map
        - does not call empty_cache() by default (expensive)
        """
        dev = get_device()

        # latents are already on dev from vae_encode(); keep them there
        cond_latents = list(cond_latents)

        # Create noises directly on the latent's device/dtype
        noises     = [torch.randn_like(latent, device=latent.device) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent, device=latent.device) for latent in cond_latents]

        # If you truly run single-process you could skip this; keeping for parity
        noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)

        # Fast path: no conditioning blur/noise
        if cond_noise_scale == 0.0:
            latent_blurs = cond_latents
        else:
            latent_blurs = []
            for lb, an in zip(cond_latents, aug_noises):
                t = torch.tensor([1000.0], device=dev).mul_(cond_noise_scale)
                shape = torch.tensor(lb.shape[1:], device=dev)[None]
                t = runner.timestep_transform(t, shape)
                latent_blurs.append(runner.schedule.forward(lb, an, t))

        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=lb)
            for noise, lb in zip(noises, latent_blurs)
        ]

        # Ensure embeds are on device (non_blocking)
        text_embeds_dict = {
            "texts_pos": [e.to(dev, non_blocking=True) for e in text_embeds_dict["texts_pos"]],
            "texts_neg": [e.to(dev, non_blocking=True) for e in text_embeds_dict["texts_neg"]],
        }

        with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=dit_offload,
                **text_embeds_dict,
            )

        samples = [
            (rearrange(v[:, None], "c t h w -> t c h w") if v.ndim == 3 else rearrange(v, "c t h w -> t c h w"))
            for v in video_tensors
        ]

        del video_tensors
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
