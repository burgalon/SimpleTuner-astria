# Copyright (c) 2023 HuggingFace Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.


from contextlib import nullcontext
from typing import Literal, Optional, Tuple, Union
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, SpatialNorm
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.downsampling import Downsample2D
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange

from astria.seedvr2.common.distributed.advanced import get_sequence_parallel_world_size
from astria.seedvr2.common.logger import get_logger
from astria.seedvr2.common.utils import safe_pad_operation, safe_interpolate_operation
from astria.seedvr2.models.video_vae_v3.modules.causal_inflation_lib import (
    InflatedCausalConv3d,
    causal_norm_wrapper,
    init_causal_conv3d,
    remove_head,
)
from astria.seedvr2.models.video_vae_v3.modules.context_parallel_lib import (
    causal_conv_gather_outputs,
    causal_conv_slice_inputs,
)
from astria.seedvr2.models.video_vae_v3.modules.global_config import set_norm_limit
from astria.seedvr2.models.video_vae_v3.modules.types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
    _receptive_field_t,
)

logger = get_logger(__name__)  # pylint: disable=invalid-name


import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


class Upsample3D(Upsample2D):
    """A 3D upsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv if self.name == "conv" else self.Conv2d_0

        assert type(conv) is not nn.ConvTranspose2d
        # Note: lora_layer is not passed into constructor in the original implementation.
        # So we make a simplification.
        conv = init_causal_conv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        self._tiling_enabled: bool = False
        self._tile_latent: int = 0            # tile size in *latent* units (H/W of z)
        self._tile_overlap: int = 0           # overlap in latent units
        self._tile_window: str = "hann"       # "hann" | "linear" | "ones"

        assert not self.interpolate
        # [Override] MAGViT v2 implementation
        if not self.interpolate:
            upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
            self.upscale_conv = nn.Conv3d(
                self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
            )
            identity = (
                torch.eye(self.channels)
                .repeat(upscale_ratio, 1)
                .reshape_as(self.upscale_conv.weight)
            )
            self.upscale_conv.weight.data.copy_(identity)
            nn.init.zeros_(self.upscale_conv.bias)

        if self.name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split([split_size, hidden_states.size(2) - split_size], dim=2)
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            hidden_states[i] = self.upscale_conv(hidden_states[i])
            hidden_states[i] = rearrange(
                hidden_states[i],
                "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                x=self.spatial_ratio,
                y=self.spatial_ratio,
                z=self.temporal_ratio,
            )

        # [Overridden] For causal temporal conv
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states, memory_state=memory_state)
            else:
                hidden_states = self.Conv2d_0(hidden_states, memory_state=memory_state)

        if not self.slicing:
            return hidden_states
        else:
            return torch.cat(hidden_states, dim=2)


class Downsample3D(Downsample2D):
    """A 3D downsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if type(conv) in [nn.Conv2d, LoRACompatibleConv]:
            # Note: lora_layer is not passed into constructor in the original implementation.
            # So we make a simplification.
            conv = init_causal_conv3d(
                self.channels,
                self.out_channels,
                kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                padding=(
                    1 if self.temporal_down else 0,
                    self.padding if self.spatial_down else 0,
                    self.padding if self.spatial_down else 0,
                ),
                inflation_mode=inflation_mode,
            )
        elif type(conv) is nn.AvgPool2d:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv and self.padding == 0 and self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = safe_pad_operation(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = init_causal_conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = init_causal_conv3d(
            self.out_channels,
            self.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        if self.up:
            assert type(self.upsample) is Upsample2D
            self.upsample = Upsample3D(
                self.in_channels,
                use_conv=False,
                inflation_mode=inflation_mode,
                slicing=slicing,
            )
        elif self.down:
            assert type(self.downsample) is Downsample2D
            self.downsample = Downsample3D(
                self.in_channels,
                use_conv=False,
                padding=1,
                name="op",
                inflation_mode=inflation_mode,
            )

        if self.use_in_shortcut:
            self.conv_shortcut = init_causal_conv3d(
                self.in_channels,
                self.conv_shortcut.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )

    def forward(
        self, input_tensor, temb, memory_state: MemoryState = MemoryState.DISABLED, **kwargs
    ):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes.
            # see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor, memory_state=memory_state)
            hidden_states = self.upsample(hidden_states, memory_state=memory_state)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, memory_state=memory_state)
            hidden_states = self.downsample(hidden_states, memory_state=memory_state)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DownEncoderBlock3D(DownEncoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
            downsample_padding=downsample_padding,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # [Override] Replace module.
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UpDecoderBlock3D(UpDecoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
            temb_channels=temb_channels,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                    slicing=slicing,
                )
            )

            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_upsample:
            # [Override] Replace module & use learnable upsample
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        interpolate=False,
                        inflation_mode=inflation_mode,
                        slicing=slicing,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            # [Override] Replace module.
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups if resnet_time_scale_shift == "default" else None
                        ),
                        spatial_norm_dim=(
                            temb_channels if resnet_time_scale_shift == "spatial" else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, memory_state: MemoryState = MemoryState.DISABLED):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state)

        return hidden_states


class Encoder3D(nn.Module):
    r"""
    [Override] override most logics to support extra condition input and causal conv

    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_down_block`
            for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        # [Override] add extra_cond_dim, temporal down num
        temporal_down_num: int = 2,
        extra_cond_dim: int = None,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.extra_cond_dim = extra_cond_dim

        self.conv_extra_cond = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # [Override] to support temporal down block design
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last ones

            assert down_block_type == "DownEncoderBlock3D"

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                # Note: Don't know why set it as 0
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

            def zero_module(module):
                # Zero out the parameters of a module and return it.
                for p in module.parameters():
                    p.detach().zero_()
                return module

            self.conv_extra_cond.append(
                zero_module(
                    nn.Conv3d(extra_cond_dim, output_channel, kernel_size=1, stride=1, padding=0)
                )
                if self.extra_cond_dim is not None and self.extra_cond_dim > 0
                else None
            )

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = init_causal_conv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    def forward(
        self,
        sample: torch.FloatTensor,
        extra_cond=None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, memory_state, use_reentrant=False
                )
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

            # sample = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.mid_block), sample, use_reentrant=False
            # )

        else:
            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = down_block(sample, memory_state=memory_state)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        # [Override] add temporal up block
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        print(f"slicing_up_num: {slicing_up_num}")
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            assert up_block_type == "UpDecoderBlock3D"
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = init_causal_conv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    # Note: Just copy from Decoder.
    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        memory_state,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, memory_state
                    )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class AutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(self, attention: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    def load_state_dict(self, state_dict, strict=True):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion. We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict)


class VideoAutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        *args,
        **kwargs,
    ):
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (2**temporal_scale_num)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # [Override] make sure it can be normally initialized
            down_block_types=tuple(
                [down_block_type.replace("3D", "2D") for down_block_type in down_block_types]
            ),
            up_block_types=tuple(
                [up_block_type.replace("3D", "2D") for up_block_type in up_block_types]
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            *args,
            **kwargs,
        )

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.slicing_encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _encode(
        self, x: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> torch.Tensor:
        _x = x.to(self.device)
        _x = causal_conv_slice_inputs(_x, self.slicing_sample_min_size, memory_state=memory_state)
        h = self.encoder(_x, memory_state=memory_state)
        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h
        output = causal_conv_gather_outputs(output)
        return output.to(x.device)

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> torch.Tensor:
        _z = z.to(self.device)
        _z = causal_conv_slice_inputs(_z, self.slicing_latent_min_size, memory_state=memory_state)
        if self.post_quant_conv is not None:
            _z = self.post_quant_conv(_z, memory_state=memory_state)
        output = self.decoder(_z, memory_state=memory_state)
        output = causal_conv_gather_outputs(output)
        return output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size * sp_size, dim=2)
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            return torch.cat(encoded_slices, dim=2)
        else:
            return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            return torch.cat(decoded_slices, dim=2)
        else:
            return self._decode(z)

    def tiled_encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Use enable_tiling
        raise NotImplementedError

    def tiled_decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # Use enable_tiling
        raise NotImplementedError

    def _encoder_spatial_factor(self) -> int:
        """Return total spatial downsample factor of the encoder."""
        factor = 1
        for m in self.encoder.modules():
            # Our Downsample3D sets spatial_down=True and stride=2 on H/W
            if isinstance(m, Downsample3D) and getattr(m, "spatial_down", False):
                # Be robust if someone tweaks stride; count only when it is 2
                s = getattr(m, "conv", None)
                if hasattr(s, "stride"):
                    st = s.stride
                    if isinstance(st, tuple) and (st[-1] == 2 or st[-2] == 2):
                        factor *= 2
                    elif st == 2:
                        factor *= 2
                else:
                    # Fallback: assume x2 if we can’t inspect
                    factor *= 2
        return factor if factor > 0 else 1

    def enable_tiling(self, *, tile_latent: int = 192, tile_overlap: int = 64, window: str = "hann", margin_l=32, **kwargs):
        """
        Enable spatial tiling for decode. Sizes are in latent grid units.
        Example: if downsample factor is 8, tile_latent=192 -> ~1536 px tiles.

        tile_latent: bigger = faster, higher OOM likelihood
        tile_overlap: smaller = faster, less memory but worse quality
        margin_l: decode context margin. Smaller margin = faster+less mem with worse quality
        """
        assert tile_latent > 0
        assert 0 <= tile_overlap < tile_latent
        assert window in ("hann", "linear", "ones")
        self._tiling_enabled = True
        self._tile_latent = int(tile_latent)
        self._tile_overlap = int(tile_overlap)
        self._tile_window = window
        self._tile_decode_margin_l = margin_l

    def disable_tiling(self):
        self._tiling_enabled = False

    @torch.no_grad()
    def _tiled_decode(self, z5: torch.Tensor) -> torch.Tensor:
        assert z5.ndim == 5, f"expected [B,C,T,Hl,Wl], got {tuple(z5.shape)}"
        B, C, T, Hl, Wl = z5.shape
        device = z5.device
        model_dtype = z5.dtype

        tile_l = int(self._tile_latent)
        ov_l   = int(self._tile_overlap)
        step_l = max(1, tile_l - ov_l)

        # If seams persist, try increasing to 96 or 128 (slower but safer).
        margin_l = self._tile_decode_margin_l

        scale_h = scale_w = None
        Ho = Wo = None
        acc, wsum = None, None

        # Use a tiny epsilon to avoid actual div-by-zero, 
        # but keep it small enough that 0-weight regions don't bleed through.
        eps = 1e-7

        def _make_edge_ramp(L: int, reverse: bool) -> torch.Tensor:
            # Standard Hann window (sine-squared) for smooth blending
            if L <= 0: return None
            t = torch.linspace(0, math.pi / 2.0, L, device=device, dtype=torch.float32)
            r = torch.sin(t) ** 2  # Ramps from 0 to 1 smoothly
            if reverse:
                r = torch.flip(r, dims=(0,))
            # Ensure we never hit true 0.0 to avoid NaNs in wsum, 
            # but it should be effectively 0 for blending.
            return r.clamp(min=eps)

        for h0 in range(0, Hl, step_l):
            for w0 in range(0, Wl, step_l):
                h1 = min(Hl, h0 + tile_l)
                w1 = min(Wl, w0 + tile_l)

                h_load_start = max(0, h0 - margin_l)
                w_load_start = max(0, w0 - margin_l)
                h_load_end = min(Hl, h1 + margin_l)
                w_load_end = min(Wl, w1 + margin_l)

                z_tile_con = z5[:, :, :, h_load_start:h_load_end, w_load_start:w_load_end]
                # Important: MemoryState.DISABLED ensures this tile is normalized independently,
                # but because it's larger than needed, the center (valid) part is stable.
                x_tile_con = self._decode(z_tile_con, memory_state=MemoryState.DISABLED)

                if scale_h is None:
                    ho_con, wo_con = x_tile_con.shape[-2:]
                    # Use round() to avoid potential float precision errors in weird VAEs
                    scale_h = int(round(ho_con / (h_load_end - h_load_start)))
                    scale_w = int(round(wo_con / (w_load_end - w_load_start)))
                    Ho, Wo = Hl * scale_h, Wl * scale_w
                    acc  = torch.zeros((B, x_tile_con.shape[1], T, Ho, Wo), device=device, dtype=torch.float32)
                    wsum = torch.zeros((1, 1, 1, Ho, Wo), device=device, dtype=torch.float32)

                h_rel_start = (h0 - h_load_start) * scale_h
                w_rel_start = (w0 - w_load_start) * scale_w
                h_rel_end = h_rel_start + (h1 - h0) * scale_h
                w_rel_end = w_rel_start + (w1 - w0) * scale_w

                x_tile = x_tile_con[..., h_rel_start:h_rel_end, w_rel_start:w_rel_end]

                oh0, ow0 = h0 * scale_h, w0 * scale_w
                oh1, ow1 = oh0 + x_tile.shape[-2], ow0 + x_tile.shape[-1]

                ov_px_h = int(ov_l * scale_h)
                ov_px_w = int(ov_l * scale_w)

                # has_prev_h = h0 > 0
                # has_next_h = h1 < Hl
                # has_prev_w = w0 > 0
                # has_next_w = w1 < Wl

                # Which neighbors exist in the latent grid?
                has_prev_h = h0 > 0
                has_next_h = h1 < Hl
                has_prev_w = w0 > 0
                has_next_w = w1 < Wl

                # One-sided halo: only add context where we haven't decoded yet.
                m_top    = 0
                m_left   = 0
                m_bottom = self._tile_decode_margin_l if has_next_h else 0
                m_right  = self._tile_decode_margin_l if has_next_w else 0

                h_load_start = max(0, h0 - m_top)
                w_load_start = max(0, w0 - m_left)
                h_load_end   = min(Hl, h1 + m_bottom)
                w_load_end   = min(Wl, w1 + m_right)

                wh = torch.ones(x_tile.shape[-2], device=device, dtype=torch.float32)
                if has_prev_h and ov_px_h > 0:
                    # Ramp UP from 0 on the top edge
                    L = min(ov_px_h, wh.numel())
                    wh[:L] *= _make_edge_ramp(L, reverse=False)
                if has_next_h and ov_px_h > 0:
                    # Ramp DOWN to 0 on the bottom edge
                    L = min(ov_px_h, wh.numel())
                    wh[-L:] *= _make_edge_ramp(L, reverse=True)

                ww = torch.ones(x_tile.shape[-1], device=device, dtype=torch.float32)
                if has_prev_w and ov_px_w > 0:
                    # Ramp UP from 0 on the left edge
                    L = min(ov_px_w, ww.numel())
                    ww[:L] *= _make_edge_ramp(L, reverse=False)
                if has_next_w and ov_px_w > 0:
                    # Ramp DOWN to 0 on the right edge
                    L = min(ov_px_w, ww.numel())
                    ww[-L:] *= _make_edge_ramp(L, reverse=True)

                w2d = wh.view(1,1,1,-1,1) * ww.view(1,1,1,1,-1)
                acc[:, :, :, oh0:oh1, ow0:ow1] += (x_tile.to(torch.float32) * w2d)
                wsum[:, :, :, oh0:oh1, ow0:ow1] += w2d

                del z_tile_con, x_tile_con, x_tile, wh, ww, w2d

        # Safe normalize
        out = acc / wsum.clamp(min=eps)
        return out.to(model_dtype)

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self,
        *args,
        spatial_downsample_factor: int,
        temporal_downsample_factor: int,
        freeze_encoder: bool,
        **kwargs,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)

    # def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
    #     with torch.no_grad() if self.freeze_encoder else nullcontext():
    #         z, p = self.encode(x)
    #     x = self.decode(z).sample
    #     return CausalAutoencoderOutput(x, z, p)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        # Keep the legacy return type & attributes expected downstream.
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            enc = self.encode(x)  # CausalEncoderOutput
        x_rec = self.decode(enc.latent).sample
        return CausalAutoencoderOutput(x_rec, enc.latent, enc.posterior)

    @torch.no_grad()
    def _tiled_encode(self, x5: torch.Tensor) -> torch.Tensor:
        assert x5.ndim == 5, f"expected [B,C,T,H,W], got {tuple(x5.shape)}"
        B, C, T, H, W = x5.shape
        device = x5.device

        tile_l = int(self._tile_latent)
        ov_l   = int(self._tile_overlap)
        step_l = max(1, tile_l - ov_l)

        f  = self._encoder_spatial_factor()
        Hl = (H + f - 1) // f
        Wl = (W + f - 1) // f

        tile_px = tile_l * f
        step_px = step_l * f

        acc  = None   # fp32
        wsum = None   # fp32
        eps  = 1e-3

        def _edge_ramp(L: int, reverse: bool, kind: str) -> torch.Tensor:
            t = torch.linspace(0, 1, L, device=device, dtype=torch.float32)
            if kind == "linear":
                r = t
            else:  # "hann" -> half-hann, 0..1
                r = 0.5 - 0.5 * torch.cos(math.pi * t)
            if reverse:
                r = torch.flip(r, dims=(0,))
            return eps + (1.0 - eps) * r  # never 0

        for h0_l in range(0, Hl, step_l):
            for w0_l in range(0, Wl, step_l):
                # slice input in pixels and right/bottom-pad
                h0_px, w0_px = h0_l * f, w0_l * f
                h1_px, w1_px = min(H, h0_px + tile_px), min(W, w0_px + tile_px)
                pad_b, pad_r = tile_px - (h1_px - h0_px), tile_px - (w1_px - w0_px)

                x_tile = x5[:, :, :, h0_px:h1_px, w0_px:w1_px]
                if (pad_b | pad_r) != 0:
                    x_tile = F.pad(x_tile, (0, pad_r, 0, pad_b), mode="constant", value=0.0)

                # encode this tile (per your existing logic)
                h_tile = self._encode(x_tile, memory_state=MemoryState.DISABLED)  # [B,Cout,T, hl_tile, wl_tile]

                if acc is None:
                    _, Cout, Tt, hl_tile, wl_tile = h_tile.shape
                    assert Tt == T
                    acc  = torch.zeros((B, Cout, T, Hl, Wl), device=device, dtype=torch.float32)
                    wsum = torch.zeros((1,   1,    1, Hl, Wl), device=device, dtype=torch.float32)

                # valid region in latent (crop off padded area)
                valid_hl = min(tile_l, Hl - h0_l)
                valid_wl = min(tile_l, Wl - w0_l)
                h_valid  = h_tile[:, :, :, :valid_hl, :valid_wl]

                # neighbor awareness in latent space
                has_prev_h = h0_l > 0
                has_next_h = (h0_l + valid_hl) < Hl
                has_prev_w = w0_l > 0
                has_next_w = (w0_l + valid_wl) < Wl

                # 1D edge-only ramps (no attenuation on outer image borders)
                wh = torch.ones(valid_hl, device=device, dtype=torch.float32)
                if ov_l > 0 and has_prev_h:
                    L = min(ov_l, valid_hl)
                    wh[:L] *= _edge_ramp(L, reverse=False, kind=self._tile_window)
                if ov_l > 0 and has_next_h:
                    L = min(ov_l, valid_hl)
                    wh[-L:] *= _edge_ramp(L, reverse=True,  kind=self._tile_window)

                ww = torch.ones(valid_wl, device=device, dtype=torch.float32)
                if ov_l > 0 and has_prev_w:
                    L = min(ov_l, valid_wl)
                    ww[:L] *= _edge_ramp(L, reverse=False, kind=self._tile_window)
                if ov_l > 0 and has_next_w:
                    L = min(ov_l, valid_wl)
                    ww[-L:] *= _edge_ramp(L, reverse=True,  kind=self._tile_window)

                w2d = wh.view(1,1,1,-1,1) * ww.view(1,1,1,1,-1)  # fp32

                acc[:, :, :, h0_l:h0_l+valid_hl, w0_l:w0_l+valid_wl] += h_valid.to(torch.float32) * w2d
                wsum[:, :, :, h0_l:h0_l+valid_hl, w0_l:w0_l+valid_wl] += w2d

                del x_tile, h_tile, h_valid, wh, ww, w2d

        h = acc / wsum.clamp_min(eps)
        return h.to(x5.dtype)

    # Old non-tiled encode
    # def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
    #     if x.ndim == 4:
    #         x = x.unsqueeze(2)
    #     p = super().encode(x).latent_dist
    #     z = p.sample().squeeze(2)
    #     return CausalEncoderOutput(z, p)

    def _encoder_spatial_factor(self) -> int:
        """Return total spatial downsample factor of the encoder."""
        factor = 1
        for m in self.encoder.modules():
            # Our Downsample3D sets spatial_down=True and stride=2 on H/W
            if isinstance(m, Downsample3D) and getattr(m, "spatial_down", False):
                # Be robust if someone tweaks stride; count only when it is 2
                s = getattr(m, "conv", None)
                if hasattr(s, "stride"):
                    st = s.stride
                    if isinstance(st, tuple) and (st[-1] == 2 or st[-2] == 2):
                        factor *= 2
                    elif st == 2:
                        factor *= 2
                else:
                    # Fallback: assume x2 if we can’t inspect
                    factor *= 2
        return factor if factor > 0 else 1

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> CausalEncoderOutput:
        x5 = x.unsqueeze(2) if x.ndim == 4 else x

        use_tiled = getattr(self, "_tiling_enabled", False) and getattr(self, "_tile_latent", 0) > 0
        if use_tiled:
            f = self._encoder_spatial_factor()
            need_tiles = ((x5.shape[-2] + f - 1) // f > self._tile_latent) or \
                        ((x5.shape[-1] + f - 1) // f > self._tile_latent)
        else:
            need_tiles = False

        if use_tiled and need_tiles:
            h = self._tiled_encode(x5)
        else:
            h = self.slicing_encode(x5)

        posterior = DiagonalGaussianDistribution(h)
        # use mean for fidelity & cross-tile consistency
        z = posterior.mode().squeeze(2)
        return CausalEncoderOutput(z, posterior)


    # Old non-tiled decode
    # def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
    #     if z.ndim == 4:
    #         z = z.unsqueeze(2)
    #     x = super().decode(z).sample.squeeze(2)
    #     return CausalDecoderOutput(x)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        # Accept [B,C,H,W] or [B,C,T,H,W]
        z5 = z.unsqueeze(2) if z.ndim == 4 else z
        use_tiled = (
            getattr(self, "_tiling_enabled", False)
            and (z5.shape[-2] > self._tile_latent or z5.shape[-1] > self._tile_latent)
        )
        if use_tiled:
            decoded = self._tiled_decode(z5)  # returns [B,C,T,H_out,W_out] on z.device
        else:
            decoded = self.slicing_decode(z5)

        if z.ndim == 4:
            decoded = decoded.squeeze(2)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = split_size // self.temporal_downsample_factor
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
