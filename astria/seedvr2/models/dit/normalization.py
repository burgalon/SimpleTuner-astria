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

from __future__ import annotations

from typing import Callable, Optional, Union, Tuple
import torch
from torch import nn

# Try native PyTorch RMSNorm (PyTorch >= 2.4/2.5)
_TORCH_RMSNORM = getattr(nn, "RMSNorm", None)

# Try Diffusers' RMSNorm (kept for backwards compat with existing envs)
try:
    from diffusers.models.normalization import RMSNorm as _DIFFUSERS_RMSNORM  # type: ignore
except Exception:
    _DIFFUSERS_RMSNORM = None  # type: ignore


NormalizedShape = Union[int, Tuple[int, ...]]
norm_layer_type = Callable[[int, float, bool], nn.Module]


class _FallbackRMSNorm(nn.Module):
    """
    Lightweight RMSNorm fallback matching common interfaces:
      - normalizes over the last len(normalized_shape) dims (LayerNorm-style)
      - supports elementwise_affine (weight and bias)
    """
    def __init__(self, normalized_shape: NormalizedShape, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            # bias is uncommon for RMSNorm, but we keep it for interface parity with some fused impls
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = x.pow(2).mean(dim=dims, keepdim=True)
        y = x * torch.rsqrt(rms + self.eps)
        if self.elementwise_affine:
            y = y * self.weight
            if self.bias is not None:
                y = y + self.bias
        return y


def _make_rmsnorm(
    dim: NormalizedShape,
    eps: float,
    elementwise_affine: bool,
) -> nn.Module:
    # Prefer native PyTorch RMSNorm if available
    if _TORCH_RMSNORM is not None:
        return _TORCH_RMSNORM(
            normalized_shape=dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    # Then try Diffusers' RMSNorm
    if _DIFFUSERS_RMSNORM is not None:
        return _DIFFUSERS_RMSNORM(
            dim=dim if isinstance(dim, int) else dim[-1],
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    # Fallback: pure PyTorch implementation
    return _FallbackRMSNorm(
        normalized_shape=dim,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:
    """
    Returns a factory that builds the requested normalization layer.

    Supported values:
      - None       -> nn.Identity
      - "layer"    -> nn.LayerNorm
      - "rms"      -> RMSNorm (torch.nn.RMSNorm if available, else diffusers, else fallback)
      - "fusedln"  -> apex FusedLayerNorm if available, else nn.LayerNorm
      - "fusedrms" -> apex FusedRMSNorm   if available, else RMSNorm fallback
    """

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool) -> nn.Module:
        if norm_type is None:
            return nn.Identity()

        nt = norm_type.lower()

        if nt == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if nt == "rms":
            return _make_rmsnorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if nt == "fusedln":
            try:
                from apex.normalization import FusedLayerNorm  # type: ignore
                return FusedLayerNorm(
                    normalized_shape=dim,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                )
            except Exception:
                # Fallback to unfused LayerNorm
                return nn.LayerNorm(
                    normalized_shape=dim,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )

        if nt == "fusedrms":
            try:
                from apex.normalization import FusedRMSNorm  # type: ignore
                return FusedRMSNorm(
                    normalized_shape=dim,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                )
            except Exception:
                # Fallback to RMSNorm
                return _make_rmsnorm(
                    dim=dim,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
