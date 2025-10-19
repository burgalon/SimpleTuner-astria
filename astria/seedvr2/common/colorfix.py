import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# --- Module-level transforms (avoid re-creating them repeatedly) ---
_TO_TENSOR = ToTensor()
_TO_IMAGE = ToPILImage()

# --- Helpers -----------------------------------------------------------------

def _select_device(user_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Choose a device. If none provided, prefer CUDA when available.
    """
    if user_device is not None:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ensure_4d_float_tensor(
    x: Union[Image.Image, Tensor],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
) -> Tensor:
    """
    Convert PIL or Tensor into a 4D float tensor in [0,1], shape (N=1, C, H, W).
    If tensor is already on device/dtype, it's reused without copy where possible.
    """
    dev = _select_device(device)
    dt = dtype or torch.float32

    if isinstance(x, Image.Image):
        t = _TO_TENSOR(x)  # (C, H, W), float32 in [0,1]
    elif torch.is_tensor(x):
        t = x
        if t.dim() == 2:  # H, W -> 1xHxW
            t = t.unsqueeze(0)
        if t.dim() == 3 and t.shape[0] not in (1, 3):  # H, W, C -> CxHxW if likely NHWC
            # attempt to fix common (H, W, C) layout
            t = t.permute(2, 0, 1).contiguous()
    else:
        raise TypeError("Input must be a PIL.Image or torch.Tensor.")

    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1, C, H, W)

    # Normalize dtype/range (assume inputs are already [0,1] if float)
    if not torch.is_floating_point(t):
        t = t.float()
        if t.max() > 1.0:
            t = t / 255.0

    # Move to device/dtype, optionally channels_last for faster conv on some GPUs
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    return t.to(device=dev, dtype=dt, memory_format=memory_format, non_blocking=True)

def _to_pil_01(t: Tensor) -> Image.Image:
    """
    Convert a (1, C, H, W) or (C, H, W) float tensor in [0,1] back to PIL.
    """
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.clamp(0.0, 1.0).cpu()
    return _TO_IMAGE(t)

# --- AdaIN -------------------------------------------------------------------

def calc_mean_std(feat: Tensor, eps: float = 1e-5) -> Tuple[Tensor, Tensor]:
    """
    Compute per-channel mean and std for a 4D tensor (N, C, H, W).
    Returns tensors with shape (N, C, 1, 1).
    """
    if feat.dim() != 4:
        raise ValueError("calc_mean_std expects a 4D tensor (N, C, H, W).")
    mean = feat.mean(dim=(2, 3), keepdim=True)
    var = feat.var(dim=(2, 3), unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std

def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor, eps: float = 1e-5) -> Tensor:
    """
    AdaIN: align content channels to style channels (per-sample, per-channel).
    Both inputs must be 4D (N, C, H, W). Channel counts must match.
    """
    if content_feat.dim() != 4 or style_feat.dim() != 4:
        raise ValueError("adaptive_instance_normalization expects 4D tensors.")
    if content_feat.shape[1] != style_feat.shape[1]:
        raise ValueError("Content and style must have the same number of channels.")

    c_mean, c_std = calc_mean_std(content_feat, eps=eps)
    s_mean, s_std = calc_mean_std(style_feat, eps=eps)

    # Normalize then re-scale/shift. Broadcasting handles shapes; no .expand needed.
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

# --- Wavelet-ish Blur/Decomposition ------------------------------------------

# Cache the base 3x3 kernel on (device, dtype) to avoid re-allocations.
# The kernel is a simple separable-ish Gaussian-like blur:
# [[1, 2, 1],
#  [2, 4, 2],
#  [1, 2, 1]] / 16
_BASE_BLUR_KERNEL = torch.tensor(
    [[0.0625, 0.125, 0.0625],
     [0.125 , 0.25 , 0.125 ],
     [0.0625, 0.125, 0.0625]],
    dtype=torch.float32,
)

def _get_base_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    return _BASE_BLUR_KERNEL.to(device=device, dtype=dtype)

def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    """
    Depthwise 3x3 blur with dilation=radius. Input shape: (N, C, H, W).
    """
    if image.dim() != 4:
        raise ValueError("wavelet_blur expects a 4D tensor (N, C, H, W).")
    n, c, h, w = image.shape
    base = _get_base_kernel(image.device, image.dtype)  # (3,3)
    # shape to (C,1,3,3) so each channel is blurred independently (depthwise conv)
    weight = base.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    pad = radius  # keeps spatial size with dilation
    x = F.pad(image, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=radius, groups=c)

def wavelet_decomposition(image: Tensor, levels: int = 3) -> Tuple[Tensor, Tensor]:
    """
    Multi-scale decomposition returning (high_freq, low_freq).
    Accumulates high-frequency residuals; low_freq is last smoothed image.
    """
    if image.dim() != 4:
        raise ValueError("wavelet_decomposition expects a 4D tensor (N, C, H, W).")

    high = torch.zeros_like(image)
    low = image
    for i in range(levels):
        radius = 2 ** i
        blurred = wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low

def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor, levels: int = 3) -> Tensor:
    """
    Give the content image the style image's low-frequency (colors/illumination),
    while preserving content high-frequency details (edges/textures).
    Shapes: both 4D, channel counts must match. Spatial sizes may differ; the
    style low-freq is resized to match content.
    """
    if content_feat.dim() != 4 or style_feat.dim() != 4:
        raise ValueError("wavelet_reconstruction expects 4D tensors.")
    if content_feat.shape[1] != style_feat.shape[1]:
        raise ValueError("Content and style must have the same number of channels.")

    with torch.no_grad():
        c_hi, _ = wavelet_decomposition(content_feat, levels=levels)
        _, s_lo = wavelet_decomposition(style_feat, levels=levels)

        # Resize style low-freq to match content spatially, if needed
        if s_lo.shape[-2:] != c_hi.shape[-2:]:
            s_lo = F.interpolate(s_lo, size=c_hi.shape[-2:], mode="bilinear", align_corners=False)

        return c_hi + s_lo

# --- User-facing wrappers (PIL in -> PIL out) --------------------------------

def adain_color_fix(
    target: Union[Image.Image, Tensor],
    source: Union[Image.Image, Tensor],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-5,
    channels_last: bool = False,
) -> Image.Image:
    """
    Apply Adaptive Instance Normalization to transfer global color/illumination
    from source to target. Returns a PIL image.
    """
    with torch.no_grad():
        tgt = _ensure_4d_float_tensor(target, device=device, dtype=dtype, channels_last=channels_last)
        src = _ensure_4d_float_tensor(source, device=tgt.device, dtype=tgt.dtype, channels_last=channels_last)
        out = adaptive_instance_normalization(tgt, src, eps=eps)
        return _to_pil_01(out)

def wavelet_color_fix(
    target: Union[Image.Image, Tensor],
    source: Union[Image.Image, Tensor],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    levels: int = 3,
    channels_last: bool = False,
) -> Image.Image:
    """
    Merge target high-frequency with source low-frequency via multi-scale blur
    to transfer overall color/tone while preserving detail. Returns a PIL image.
    """
    with torch.no_grad():
        tgt = _ensure_4d_float_tensor(target, device=device, dtype=dtype, channels_last=channels_last)
        src = _ensure_4d_float_tensor(source, device=tgt.device, dtype=tgt.dtype, channels_last=channels_last)
        out = wavelet_reconstruction(tgt, src, levels=levels)
        return _to_pil_01(out)
