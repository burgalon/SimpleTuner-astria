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

import math
from contextlib import contextmanager
from typing import List, Optional, Union
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d

from astria.seedvr2.common.distributed.advanced import (
    get_next_sequence_parallel_rank,
    get_prev_sequence_parallel_rank,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from astria.seedvr2.common.utils import safe_pad_operation
from astria.seedvr2.common.logger import get_logger
from astria.seedvr2.models.video_vae_v3.modules.context_parallel_lib import (
    cache_send_recv,
    get_cache_size,
    _sp_is_active,  # reuse the single-GPU detector
)
from astria.seedvr2.models.video_vae_v3.modules.global_config import get_norm_limit
from astria.seedvr2.models.video_vae_v3.modules.types import MemoryState, _inflation_mode_t, _memory_device_t

logger = get_logger(__name__)


@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding


def _out_len_1d(in_len: int, k: int, s: int, d: int, p: int) -> int:
    # PyTorch conv formula
    return (in_len + 2 * p - d * (k - 1) - 1) // s + 1

def _conv3d_out_shape(sz_bcthw, module: nn.Conv3d) -> tuple:
    # sz_bcthw = (B, C_in, T, H, W) AFTER padding+concat (what you already compute)
    b, _, t, h, w = sz_bcthw
    kt, kh, kw   = module.kernel_size
    st, sh, sw   = module.stride
    dt, dh, dw   = module.dilation
    # module.padding is *spatial* here; temporal pad is handled causally
    pt, ph, pw   = module.padding  # note: for this class, padding[0] has been zeroed in __init__
    t_out = _out_len_1d(t, kt, st, dt, pt)
    h_out = _out_len_1d(h, kh, sh, dh, ph)
    w_out = _out_len_1d(w, kw, sw, dw, pw)
    return (b, module.out_channels, t_out, h_out, w_out)


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.
        self.memory_limit = float("inf")

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),  # (w_l, w_r, h_l, h_r, t_l, t_r)
        prev_cache=None,
    ):
        # Fast path: no limit → concat (& let Conv3d handle its own padding)
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            x = x.contiguous(memory_format=torch.channels_last_3d)
            return super().forward(x)

        # Compute post-concat, post-pad shape in pure Python (no tensor allocs)
        sz = list(x.size())  # [B, C, T, H, W]
        if prev_cache is not None:
            sz[split_dim - 1] += prev_cache.size(split_dim - 1)
        pw_l, pw_r, ph_l, ph_r, pt_l, pt_r = padding
        sz[-1] += (pw_l + pw_r)  # W
        sz[-2] += (ph_l + ph_r)  # H
        sz[-3] += (pt_l + pt_r)  # T

        numel = 1
        for d in sz:
            numel *= int(d)
        # memory_occupy = (numel * x.element_size()) / (1024.0 ** 3)  # GiB
        out_bcthw = _conv3d_out_shape(sz, self)
        out_numel = 1
        for d in out_bcthw:
            out_numel *= int(d)
        memory_occupy = (out_numel * self.weight.element_size()) / (1024.0 ** 3)  # GiB

        logger.debug(
            f"x:({tuple(sz)}, {x.dtype}) {memory_occupy:.3f}GiB "
            f"prev_cache:{tuple(prev_cache.shape) if prev_cache is not None else None}"
        )

        # If it fits (or we're at the last split dimension), just do it
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            if (pw_l | pw_r | ph_l | ph_r | pt_l | pt_r) != 0:
                # constant mode is fast; no need for the safe wrapper here
                x = F.pad(x, padding, mode="constant", value=0.0)
            with ignore_padding(self):
                x = x.contiguous(memory_format=torch.channels_last_3d)
                return super().forward(x)

        logger.debug(
            f"Exceed memory limit {memory_occupy:.3f} > {self.memory_limit:.3f}, split dim {split_dim}"
        )

        # Split x (and prev_cache) along the target dim
        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x_slices = list(torch.split(x, split_sizes, dim=split_dim))
        logger.debug(f"Conv inputs: {[tuple(inp.size()) for inp in x_slices]} {x_slices[0].dtype}")
        if prev_cache is not None:
            prev_cache_slices = list(torch.split(prev_cache, split_sizes, dim=split_dim))
        else:
            prev_cache_slices = [None] * len(x_slices)

        # Recursive forward over slices
        cache = None
        for idx in range(len(x_slices)):
            cur = x_slices[idx]
            prev_cur = prev_cache_slices[idx]

            # Concat incoming cache on the previous dimension
            if prev_cur is not None:
                cur = torch.cat([prev_cur, cur], dim=split_dim - 1)

            # Build per-slice asymmetric padding (only pad at outer edges)
            # Indexing into (w_l,w_r,h_l,h_r,t_l,t_r)
            lpad_dim = (cur.ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            p = list(padding)
            p[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            p[rpad_dim] = self.padding[split_dim - 2] if idx == len(x_slices) - 1 else 0
            pad_len = p[lpad_dim] + p[rpad_dim]
            p = tuple(p)

            # Prepare next cache length (for this split dimension)
            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=cur.size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= cur.size(split_dim)
                next_cache = cur.transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)

            # Recurse into the next split dimension
            out = self.memory_limit_conv(
                cur,
                split_dim=split_dim + 1,
                padding=p,
                prev_cache=cache,
            )

            x_slices[idx] = out
            cache = next_cache

        logger.debug(f"Conv outputs, concat(dim={split_dim}): {[tuple(d.size()) for d in x_slices]}")
        return torch.cat(x_slices, dim=split_dim)

    def forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        # Single-GPU fast path: no SP and no splitting → use basic_forward.
        if math.isinf(self.memory_limit) and torch.is_tensor(input) and not _sp_is_active():
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]

        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)

        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )

        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
            # Avoid expensive sync; only move to CPU if explicitly requested
            if self.memory_device == "cpu" and self.memory is not None:
                try:
                    self.memory = self.memory.to("cpu", non_blocking=True)
                except Exception:
                    self.memory = self.memory.to("cpu")

        # Feed CuDNN with a cache/layer-friendly layout
        input = input.contiguous(memory_format=torch.channels_last_3d)
        return super().forward(input)

    def slicing_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2
        )

        # Discover SP only when active; otherwise use safe single-GPU defaults.
        sp_active = _sp_is_active()
        if sp_active:
            sp_rank = get_sequence_parallel_rank()
            sp_size = get_sequence_parallel_world_size()
            sp_group = get_sequence_parallel_group()
            send_dst = get_next_sequence_parallel_rank()
            recv_src = get_prev_sequence_parallel_rank()
        else:
            sp_rank = 0
            sp_size = 1
            sp_group = None
            send_dst = None
            recv_src = None

        # For slice=4 and sp=2, and 17 frames in total
        #                  sp0                  sp1
        # slice 0: [`0 0` 0 1 2 {3 4}]   [{3 4} 5 6 (7 8)]    extend=`0 0` cache={3 4} memory=(7 8)
        # slice 1: [(7 8) 9 10 {11 12}]  [{11 12} 13 14 15 16]
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and sp_rank in [0, sp_size - 1]
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            assert cache_size <= input[-1].size(2)
            if sp_size == 1:
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()
            else:
                if sp_rank == sp_size - 1:
                    dist.send(
                        input[-1][:, :, -cache_size:].detach().contiguous(),
                        send_dst,
                        group=sp_group,
                    )
                if sp_rank == 0:
                    shape = list(input[0].size())
                    shape[2] = cache_size
                    self.memory = torch.empty(
                        *shape, device=input[0].device, dtype=input[0].dtype
                    ).contiguous()
                    dist.recv(self.memory, recv_src, group=sp_group)
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache,
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

    def tflops(self, args, kwargs, output) -> float:
        if torch.is_tensor(output):
            output_numel = output.numel()
        elif isinstance(output, list):
            output_numel = sum(o.numel() for o in output)
        else:
            raise NotImplementedError
        return (2 * math.prod(self.kernel_size) * self.in_channels * (output_numel / 1e6)) / 1e6

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=inflate_weight,
                inflate_bias_fn=inflate_bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize while preserving the incoming dtype. Only cast back to the
    input dtype when the norm changes it. Avoids redundant .to(...) hops.
    """
    input_dtype = x.dtype

    # LayerNorm / RMSNorm expect channels-last; temporarily permute.
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x_ = rearrange(x, "b c h w -> b h w c")
            y = norm_layer(x_)
            y = rearrange(y, "b h w c -> b c h w")
            if y.dtype != input_dtype:
                y = y.to(input_dtype, non_blocking=True)
            return y

        if x.ndim == 5:
            x_ = rearrange(x, "b c t h w -> b t h w c")
            y = norm_layer(x_)
            y = rearrange(y, "b t h w c -> b c t h w")
            if y.dtype != input_dtype:
                y = y.to(input_dtype, non_blocking=True)
            return y

    # GroupNorm / BatchNorm paths.
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        # 2D/3D (<=4D tensor) — apply directly.
        if x.ndim <= 4:
            y = norm_layer(x)
            if y.dtype != input_dtype:
                y = y.to(input_dtype, non_blocking=True)
            return y

        # 3D video (5D tensor): flatten time for norm, then restore.
        if x.ndim == 5:
            t = x.size(2)
            x2 = rearrange(x, "b c t h w -> (b t) c h w")

            # If very large GroupNorm activation, chunk to cap memory.
            memory_occupy_gib = x2.numel() * x2.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy_gib > get_norm_limit():
                num_chunks = min(4 if x2.element_size() == 2 else 2, norm_layer.num_groups)
                assert norm_layer.num_groups % num_chunks == 0
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                xs = list(x2.chunk(num_chunks, dim=1))
                ws = norm_layer.weight.chunk(num_chunks, dim=0)
                bs = norm_layer.bias.chunk(num_chunks, dim=0)

                for i, (w, b) in enumerate(zip(ws, bs)):
                    xi = F.group_norm(xs[i], num_groups_per_chunk, w, b, norm_layer.eps)
                    if xi.dtype != input_dtype:
                        xi = xi.to(input_dtype, non_blocking=True)
                    xs[i] = xi
                y2 = torch.cat(xs, dim=1)
            else:
                y2 = norm_layer(x2)

            y = rearrange(y2, "(b t) c h w -> b c t h w", t=t)
            if y.dtype != input_dtype:
                y = y.to(input_dtype, non_blocking=True)
            return y

    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    # Single-GPU safe: only rank 0 does removal when SP inactive.
    sp_rank = get_sequence_parallel_rank() if _sp_is_active() else 0
    if times == 0 or sp_rank > 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        inflation_mode: Placeholder to align `inflate_weight`.
    """
    assert bias_3d.shape == bias_2d.shape
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict
