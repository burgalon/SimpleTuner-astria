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
SeedVR2 7B – Image Upscaling API (split VAE / Transformer)

- For SeedVR2 SR, DiT returns *pixel-space* tensors in [-1,1] (T,C,H,W). You usually
  do NOT VAE-decode that output; just convert it to an image (optionally color-match).
"""

from __future__ import annotations

import os
import gc
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List

import torch
import numpy as np
from PIL import Image

from einops import rearrange
from omegaconf import OmegaConf

from astria.astria_utils import CACHE_DIR
from astria.tensorize import seedvr2_load_or_tensorize
from astria.seedvr2.download import ensure_seedvr2_7b_checkpoint, ensure_seedvr2_vae_checkpoint

# Optional for saving – same dep used in the original scripts
try:
    import mediapy  # type: ignore
except Exception:
    mediapy = None  # saving still possible via numpy return

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
    def _normalize(p: Union[str, Path]) -> Path:
        p = Path(p)
        if p.is_dir():
            p = p / "main.yaml"
        return p

    if explicit is not None:
        p = _normalize(explicit)
        if p.is_file():
            return p.resolve()

    env_val = os.getenv("SEEDVR2_CONFIG_PATH")
    if env_val:
        p = _normalize(env_val)
        if p.is_file():
            return p.resolve()

    here = Path(__file__).resolve().parent
    candidate = here / "configs_7b" / "main.yaml"
    if candidate.is_file():
        return candidate

    candidate = Path.cwd() / "configs_7b" / "main.yaml"
    if candidate.is_file():
        return candidate.resolve()

    try:
        from importlib.resources import files as ir_files  # py>=3.9
        pkg_candidate = ir_files("astria.seedvr2") / "configs_7b" / "main.yaml"
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


def _cuda_empty_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cpu_hist_match_uint8(src_hwc_u8: np.ndarray, ref_hwc_u8: np.ndarray) -> np.ndarray:
    if src_hwc_u8.dtype != np.uint8 or ref_hwc_u8.dtype != np.uint8:
        raise ValueError("Expected uint8 arrays.")
    if src_hwc_u8.ndim != 3 or ref_hwc_u8.ndim != 3 or src_hwc_u8.shape[2] != 3 or ref_hwc_u8.shape[2] != 3:
        raise ValueError("Expected HxWx3 arrays.")
    out = np.empty_like(src_hwc_u8)
    for c in range(3):
        src = src_hwc_u8[..., c].ravel()
        ref = ref_hwc_u8[..., c].ravel()
        s_values, s_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
        r_values, r_counts = np.unique(ref, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64); s_quantiles /= s_quantiles[-1]
        r_quantiles = np.cumsum(r_counts).astype(np.float64); r_quantiles /= r_quantiles[-1]
        interp_vals = np.interp(s_quantiles, r_quantiles, r_values)
        out[..., c] = interp_vals[s_idx].reshape(src_hwc_u8.shape[:2]).astype(np.uint8)
    return out


def _gpu_hist_match_uint8(src_hwc_u8: torch.Tensor,
                          ref_hwc_u8: torch.Tensor,
                          downsample: int = 4) -> torch.Tensor:
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
    if isinstance(img, str):
        if not _is_image_file(img):
            raise ValueError(f"Unsupported path or extension for image: {img}")
        t = read_image(img)  # (C,H,W) uint8
        t = t.unsqueeze(0).float() / 255.0  # (1,C,H,W)
        return t

    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img = np.array(img)  # HWC uint8

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"NumPy image must be HxWxC or CxHxW; got shape {img.shape}")
        if arr.shape[0] in {1, 3} and arr.shape[2] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.clip(arr, 0.0, 1.0)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return torch.from_numpy(arr)

    if torch.is_tensor(img):
        t = img
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] not in {1, 3} and t.shape[2] in {1, 3}:
                t = t.permute(2, 0, 1)
            t = t.unsqueeze(0)
        elif t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"Expected a single image (T=1), got tensor shape {tuple(t.shape)}")
        else:
            raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}")

        t = t.float()
        if t.max() > 1.0 or t.min() < 0.0:
            if t.dtype.is_floating_point:
                t = torch.clamp(t, 0, 255)
            t = t / 255.0
        return t

    raise TypeError(f"Unsupported image input type: {type(img)}")


# ------------------------------ Main API ----------------------------------------

class SeedVR2ImageUpscaler:
    """
    Image upscaling helper around SeedVR2 7B (split VAE/DiT lifecycle).
    """
    from astria.seedvr2.download import DEFAULT_FILENAME_SHARP

    # --------- Construction ---------
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

        # Paths filled during _ensure_runner()
        self._ckpt_path: Optional[str] = None
        self.pos_path: Optional[str] = None
        self.neg_path: Optional[str] = None
        self._vae_ckpt_path: Optional[str] = None

        self._video_transform = Compose(
            [
                NaResize(
                    resolution=(self.res_h * self.res_w) ** 0.5,
                    mode="area",
                    downsample_only=False,
                ),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                VidRearrange("t c h w -> c t h w"),
            ]
        )

        self._torch_inited: bool = False

    # --------- Internal helpers ---------
    def _ensure_torch(self) -> None:
        if self._torch_inited:
            return
        import datetime
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self._torch_inited = True

    def _ensure_runner(self) -> VideoDiffusionInfer:
        if self.runner is not None:
            return self.runner

        self._ensure_torch()
        config = load_config(str(self.config_path))
        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)

        here = Path(__file__).resolve().parent
        vae_yaml = here / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml"
        if vae_yaml.is_file():
            runner.config.vae.model.__inherit__ = str(vae_yaml)

        vae_ckpt_path = ensure_seedvr2_vae_checkpoint(target_dir=str(CACHE_DIR))
        ckpt_path, pos_path, neg_path = ensure_seedvr2_7b_checkpoint(self.checkpoint_path)

        self._vae_ckpt_path = str(vae_ckpt_path)
        self._ckpt_path = str(ckpt_path)
        self.pos_path = pos_path
        self.neg_path = neg_path

        self.runner = runner
        return runner

    def _dit_loaded(self) -> bool:
        return (self.runner is not None) and hasattr(self.runner, "dit") and (self.runner.dit is not None)

    def _vae_loaded(self) -> bool:
        return (self.runner is not None) and hasattr(self.runner, "vae") and (self.runner.vae is not None)

    def _flex_configure(self, fn, *, checkpoint: Optional[str], device: Optional[str]) -> None:
        """
        Call a configure_* function with signature differences handled across forks.
        Tries, in order:
            (device=..., checkpoint=...), (checkpoint=...),
            (checkpoint as positional), (), then moves module to device externally.
        """
        # Some forks read the path from config, so set it if present.
        try:
            if self.runner and checkpoint:
                # VAE path
                if hasattr(self.runner.config, "vae") and hasattr(self.runner.config.vae, "checkpoint"):
                    self.runner.config.vae.checkpoint = checkpoint  # type: ignore[attr-defined]
                # DiT path
                if hasattr(self.runner.config, "dit") and hasattr(self.runner.config.dit, "checkpoint"):
                    self.runner.config.dit.checkpoint = checkpoint  # type: ignore[attr-defined]
        except Exception:
            pass

        tried: List[str] = []
        for attempt in (
            lambda: fn(device=device, checkpoint=checkpoint),
            lambda: fn(checkpoint=checkpoint),
            lambda: fn(checkpoint),  # positional
            lambda: fn(),
        ):
            try:
                attempt()
                return
            except TypeError as e:
                tried.append(str(e))
                continue
        # If we got here, just call without args; if that fails for non-TypeError, raise it
        fn()

    # --------- Loading / Unloading ---------
    def load_transformer(self, *, fast_tensorizer: bool = False) -> None:
        runner = self._ensure_runner()
        if self._dit_loaded():
            return

        if fast_tensorizer:
            # Fast path may instantiate both DiT and VAE; drop VAE immediately.
            dit, vae = seedvr2_load_or_tensorize(
                config_yaml=str(self.config_path),
                dit_ckpt=str(self._ckpt_path),
                vae_ckpt=str(self._vae_ckpt_path),
                device=self._device,
                force_rebuild=False,
            )
            dit.set_gradient_checkpointing(False)
            runner.dit = dit
            try:
                del vae
            except Exception:
                pass
            gc.collect()
            _cuda_empty_cache()
        else:
            self._flex_configure(runner.configure_dit_model, checkpoint=self._ckpt_path, device=self._device)
            # Post-config, ensure device & knobs
            try:
                runner.dit.to(self._device)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                runner.config.dit.gradient_checkpoint = False  # type: ignore[attr-defined]
                runner.dit.set_gradient_checkpointing(False)    # type: ignore[attr-defined]
            except Exception:
                pass

    def unload_transformer(self) -> None:
        if self.runner is None:
            return
        try:
            if hasattr(self.runner, "dit"):
                self.runner.dit = None  # type: ignore
        except Exception:
            pass
        gc.collect()
        _cuda_empty_cache()

    def load_vae(
        self,
        *,
        vae_decode_conv_max_mem: float = 8.0,
        vae_decode_norm_max_mem: float = 4.0,
        causal_slicing: bool = False,
        causal_slicing_split_size: int = 0,
        causal_slicing_memory_device: str = 'cpu',
        tiling: bool = False,
    ) -> None:
        runner = self._ensure_runner()
        if self._vae_loaded():
            return

        # Flexible configure to tolerate signature changes
        self._flex_configure(runner.configure_vae_model, checkpoint=self._vae_ckpt_path, device=self._device)

        vae = runner.vae
        vae.requires_grad_(False).eval()

        # Optional: slicing/memory knobs
        try:
            if not causal_slicing and hasattr(runner.config, "vae") and hasattr(runner.config.vae, "slicing") and hasattr(vae, "set_causal_slicing"):
                vae.set_causal_slicing(**runner.config.vae.slicing)
            if causal_slicing and hasattr(vae, "set_causal_slicing"):
                vae.set_causal_slicing(**{
                    'split_size': causal_slicing_split_size,
                    'memory_device': causal_slicing_memory_device,
                })
        except Exception:
            pass

        try:
            if hasattr(vae, "set_memory_limit"):
                vae.set_memory_limit(
                    conv_max_mem=vae_decode_conv_max_mem,
                    norm_max_mem=vae_decode_norm_max_mem,
                )
        except Exception:
            pass

        # Keep special convs on same device for performance (if available in this fork)
        if not causal_slicing:
            try:
                from astria.seedvr2.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
                from astria.seedvr2.models.video_vae_v3.modules.inflated_layers import InflatedCausalConv3d as InflatedCausalConv3dV2
                for m in vae.modules():
                    if isinstance(m, InflatedCausalConv3d) or isinstance(m, InflatedCausalConv3dV2):
                        m.set_memory_device("same")
            except Exception:
                pass

        if tiling:
            vae.enable_tiling()

    def unload_vae(self) -> None:
        if self.runner is None:
            return
        try:
            if hasattr(self.runner, "vae"):
                self.runner.vae = None  # type: ignore
        except Exception:
            pass
        gc.collect()
        _cuda_empty_cache()

    def load(
        self,
        vae_decode_conv_max_mem: float = 8.0,
        vae_decode_norm_max_mem: float = 4.0,
        causal_slicing: bool = False,
        causal_slicing_split_size: int = 0,
        causal_slicing_memory_device: str = 'cpu',
    ) -> None:
        self.load_transformer(fast_tensorizer=False)
        self.load_vae(
            vae_decode_conv_max_mem=vae_decode_conv_max_mem,
            vae_decode_norm_max_mem=vae_decode_norm_max_mem,
            causal_slicing=causal_slicing,
            causal_slicing_split_size=causal_slicing_split_size,
            causal_slicing_memory_device=causal_slicing_memory_device,
        )

    # --------- Text embeddings ---------
    def set_text_embeddings(self, pos: torch.Tensor, neg: torch.Tensor) -> None:
        self._text_pos = pos
        self._text_neg = neg

    def clear_text_embeddings(self) -> None:
        self._text_pos = None
        self._text_neg = None

    # --------- Split steps ---------
    @torch.inference_mode()
    def encode_image_to_cond_latent(
        self,
        image: Union[str, "np.ndarray", "torch.Tensor", "Image.Image"],
        *,
        model_offloading: bool = False,
    ) -> torch.Tensor:
        if self.runner is None:
            raise RuntimeError("Model not initialized. Call load_vae() (or load()) first.")
        if not self._vae_loaded():
            raise RuntimeError("VAE not loaded. Call load_vae() first.")

        tchw = _to_tchw(image)  # (1,C,H,W)
        cond = self._video_transform(tchw.to(self._device))  # C T H W

        if model_offloading and self._dit_loaded():
            self.runner.dit.to("cpu")
            self.runner.vae.to(self._device)

        latents_list: List[torch.Tensor] = self.runner.vae_encode([cond])  # list of latents
        cond_latent = latents_list[0]

        if model_offloading and self._dit_loaded():
            self.runner.vae.to("cpu")
            self.runner.dit.to(self._device)

        return cond_latent

    @torch.inference_mode()
    def upscale_to_latent(
        self,
        cond_latent: torch.Tensor,
        *,
        seed: int = 666,
        sample_steps: int = 1,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        dit_offload: bool = False,
        pos_emb_path: Optional[str] = None,
        neg_emb_path: Optional[str] = None,
    ) -> torch.Tensor:
        if self.runner is None:
            raise RuntimeError("Model not initialized. Call load_transformer() (or load()) first.")
        if not self._dit_loaded():
            raise RuntimeError("Transformer not loaded. Call load_transformer() first.")

        set_seed(seed, same_across_ranks=True)
        self.runner.config.diffusion.cfg.scale = float(cfg_scale)
        self.runner.config.diffusion.cfg.rescale = float(cfg_rescale)
        self.runner.config.diffusion.timesteps.sampling.steps = int(sample_steps)
        self.runner.configure_diffusion()

        texts_pos, texts_neg = self._resolve_text_embeds(pos_emb_path, neg_emb_path)
        text_embeds_dict = {"texts_pos": [texts_pos], "texts_neg": [texts_neg]}

        cond_latents = [cond_latent]
        noises     = [torch.randn_like(latent, device=latent.device) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent, device=latent.device) for latent in cond_latents]
        noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)

        # Fast path: no conditioning blur/noise
        latent_blurs = cond_latents

        conditions = [
            self.runner.get_condition(noise, task="sr", latent_blur=lb)
            for noise, lb in zip(noises, latent_blurs)
        ]

        dev = get_device()
        text_embeds_dict = {
            "texts_pos": [e.to(dev, non_blocking=True) for e in text_embeds_dict["texts_pos"]],
            "texts_neg": [e.to(dev, non_blocking=True) for e in text_embeds_dict["texts_neg"]],
        }

        with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=True):
            video_tensors = self.runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=dit_offload,
                **text_embeds_dict,
            )

        out = video_tensors[0]
        if out.ndim == 3:
            out = rearrange(out[:, None], "c t h w -> t c h w")  # -> 1,C,H,W
        else:
            out = rearrange(out, "c t h w -> t c h w")
        return out

    @torch.inference_mode()
    def upscale_to_vae_latent(
        self,
        cond_latent: torch.Tensor,
        *,
        seed: int = 666,
        sample_steps: int = 1,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        dit_offload: bool = False,
        pos_emb_path: Optional[str] = None,
        neg_emb_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        DiT step ONLY for SR: returns a *VAE latent* (no decode).
        Requires: transformer loaded; VAE is NOT required.
        Output: latent with shape like VAE space (..., C) i.e., (T,H,W,C).
        """
        if self.runner is None:
            raise RuntimeError("Model not initialized. Call load_transformer() (or load()) first.")
        if not self._dit_loaded():
            raise RuntimeError("Transformer not loaded. Call load_transformer() first.")

        # Configure diffusion knobs
        set_seed(seed, same_across_ranks=True)
        self.runner.config.diffusion.cfg.scale = float(cfg_scale)
        self.runner.config.diffusion.cfg.rescale = float(cfg_rescale)
        self.runner.config.diffusion.timesteps.sampling.steps = int(sample_steps)
        self.runner.configure_diffusion()

        # Text embeds (tensors)
        texts_pos, texts_neg = self._resolve_text_embeds(pos_emb_path, neg_emb_path)

        # Build noises/conditions (same shapes as in infer.py)
        noises     = [torch.randn_like(cond_latent, device=cond_latent.device)]
        aug_noises = [torch.randn_like(cond_latent, device=cond_latent.device)]
        noises, aug_noises, cond_latents = sync_data((noises, aug_noises, [cond_latent]), 0)
        latent_blurs = cond_latents  # no extra blur

        conditions = [
            self.runner.get_condition(noise, task="sr", latent_blur=lb)
            for noise, lb in zip(noises, latent_blurs)
        ]

        # Flatten like VideoDiffusionInfer.inference()
        try:
            from astria.seedvr2.models.dit_v2 import na as nav2
        except Exception:
            from astria.seedvr2.models.dit import na as nav2
        from astria.seedvr2.common.diffusion import classifier_free_guidance_dispatcher

        batch_size = len(noises)
        # Text flatten
        text_pos_embeds, text_pos_shapes = nav2.flatten([texts_pos])
        text_neg_embeds, text_neg_shapes = nav2.flatten([texts_neg])
        # Latent + condition flatten
        latents, latents_shapes   = nav2.flatten(noises)
        latents_cond, _           = nav2.flatten(conditions)

        # Sample with DiT
        was_training = self.runner.dit.training
        self.runner.dit.eval()
        latents = self.runner.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.runner.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.runner.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=cfg_scale,
                rescale=self.runner.config.diffusion.cfg.rescale,
            ),
        )
        self.runner.dit.train(was_training)

        # Unflatten -> list of latents (T,H,W,C); we return the first
        latents = nav2.unflatten(latents, latents_shapes)

        if dit_offload:
            self.runner.dit.to("cpu")

        return latents[0]

    @torch.inference_mode()
    def vae_decode_to_pil(
        self,
        latent: torch.Tensor,
        *,
        ref_image: Optional[Union[str, np.ndarray, torch.Tensor, Image.Image]] = None,
        slow_color_fix: bool = False,
    ) -> Image.Image:
        if self.runner is None:
            raise RuntimeError("Model not initialized. Call load_vae() (or load()) first.")
        if not self._vae_loaded():
            raise RuntimeError("VAE not loaded. Call load_vae() first.")

        if hasattr(self.runner, "vae_decode"):
            decoded_list = self.runner.vae_decode([latent])  # type: ignore[attr-defined]
            decoded = decoded_list[0]
        else:
            if not hasattr(self.runner, "vae") or not hasattr(self.runner.vae, "decode"):
                raise RuntimeError("VAE decode method not found on runner.")
            decoded = self.runner.vae.decode(latent)  # type: ignore[attr-defined]

        if decoded.ndim == 3:
            decoded_tchw = decoded[None, ...]  # 1,C,H,W
        elif decoded.ndim == 4:
            if decoded.shape[0] in (1, 3, 4) and decoded.shape[1] not in (1, 3, 4):
                decoded_tchw = rearrange(decoded, "c t h w -> t c h w")
            else:
                decoded_tchw = decoded  # already T,C,H,W
        else:
            raise ValueError(f"Unexpected decoded shape: {tuple(decoded.shape)}")

        return self.decode_model_output_to_pil(decoded_tchw, ref_image=ref_image, slow_color_fix=slow_color_fix)

    @torch.inference_mode()
    def decode_model_output_to_pil(
        self,
        model_output_tchw: torch.Tensor,
        *,
        ref_image: Optional[Union[str, np.ndarray, torch.Tensor, Image.Image]] = None,
        slow_color_fix: bool = False,
    ) -> Image.Image:
        if model_output_tchw.ndim != 4 or model_output_tchw.shape[0] != 1:
            raise ValueError(f"Expected T,C,H,W with T=1; got {tuple(model_output_tchw.shape)}")

        out_chw = model_output_tchw[0]
        out_hwc_u8 = (
            rearrange(out_chw, "c h w -> h w c")
            .clamp_(-1, 1).mul_(0.5).add_(0.5).mul_(255).round_()
            .to(torch.uint8)
        )

        if ref_image is None and not slow_color_fix:
            return self.as_pil(out_hwc_u8)

        ref_tchw = _to_tchw(ref_image) if ref_image is not None else None
        if slow_color_fix:
            inp = None
            if ref_tchw is not None:
                ref_cond = self._video_transform(ref_tchw.to(out_hwc_u8.device if out_hwc_u8.is_cuda else "cpu"))
                inp = rearrange(ref_cond, "c t h w -> t c h w")
            sample = rearrange(out_hwc_u8.float().div(127.5).sub(1.0), "h w c -> 1 c h w")
            try:
                wave = wavelet_reconstruction(sample.to("cpu"), inp[:1].to("cpu") if inp is not None else sample.to("cpu"))
                hwc = rearrange(wave[0], "c h w -> h w c")
                final_u8 = (
                    hwc.clamp_(-1, 1).mul_(0.5).add_(0.5).mul_(255).round_()
                    .to(torch.uint8).cpu().numpy()
                )
                return self.as_pil(final_u8)
            except Exception:
                warnings.warn("Color fix failed; returning raw output.")
                return self.as_pil(out_hwc_u8)

        if out_hwc_u8.is_cuda and ref_tchw is not None:
            ref_hwc_u8_gpu = (
                rearrange(ref_tchw.to(out_hwc_u8.device)[0], "c h w -> h w c")
                .clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            )
            matched_u8 = _gpu_hist_match_uint8(out_hwc_u8, ref_hwc_u8_gpu, downsample=4).cpu().numpy()
            return self.as_pil(matched_u8)
        else:
            out_np = out_hwc_u8.detach().cpu().numpy()
            if ref_tchw is None:
                return self.as_pil(out_np)
            ref_np = (
                rearrange(ref_tchw[0], "c h w -> h w c")
                .clamp_(0, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()
            )
            matched = _cpu_hist_match_uint8(out_np, ref_np)
            return self.as_pil(matched)

    # --------- End-to-end ---------
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
        if self.runner is None:
            self.load()
        else:
            if not self._vae_loaded():
                self.load_vae()
            if not self._dit_loaded():
                self.load_transformer()

        # Keep a CPU reference for optional histogram color fix.
        ref_uint8 = None
        try:
            ref_uint8 = (
                _to_tchw(image)[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
        except Exception:
            pass

        cond_latent = self.encode_image_to_cond_latent(image, model_offloading=model_offloading)
        model_out_tchw = self.upscale_to_latent(
            cond_latent,
            seed=seed,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            dit_offload=dit_offload,
            pos_emb_path=pos_emb_path,
            neg_emb_path=neg_emb_path,
        )

        pil_img = self.decode_model_output_to_pil(model_out_tchw, ref_image=ref_uint8, slow_color_fix=slow_color_fix)

        if return_pil and return_torch:
            raise ValueError("Choose only one of return_pil or return_torch.")

        if return_pil:
            return pil_img

        arr = np.array(pil_img)
        if return_torch:
            return torch.from_numpy(arr)
        return arr

    # --------- Embedding resolver ---------
    def _resolve_text_embeds(
        self,
        pos_emb_path: Optional[str],
        neg_emb_path: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._text_pos is not None and self._text_neg is not None:
            return self._text_pos.to(self._device), self._text_neg.to(self._device)

        if pos_emb_path is not None and neg_emb_path is not None:
            pos = torch.load(pos_emb_path, map_location=self._device)
            neg = torch.load(neg_emb_path, map_location=self._device)
            return pos, neg

        cwd_pos, cwd_neg = "pos_emb.pt", "neg_emb.pt"
        if os.path.exists(cwd_pos) and os.path.exists(cwd_neg):
            pos = torch.load(cwd_pos, map_location=self._device)
            neg = torch.load(cwd_neg, map_location=self._device)
            return pos, neg

        if self.pos_path and self.neg_path and os.path.exists(self.pos_path) and os.path.exists(self.neg_path):
            pos = torch.load(self.pos_path, map_location=self._device)
            neg = torch.load(self.neg_path, map_location=self._device)
            return pos, neg

        raise RuntimeError(
            "Text embeddings not provided. Call `set_text_embeddings(pos, neg)` or pass "
            "`pos_emb_path` and `neg_emb_path` to `upscale_to_latent(...)`/`upscale(...)`."
        )

    # --------- I/O convenience ---------
    @staticmethod
    def save_image(img: Union["np.ndarray", "torch.Tensor"], path: str) -> None:
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
