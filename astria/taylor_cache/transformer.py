# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import types

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin, FluxTransformer2DLoadersMixin
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxTransformer2DModel,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import List


from astria.taylor_cache.forwards import (
    taylorseer_flux_single_block_forward, 
    taylorseer_flux_double_block_forward, 
    taylorseer_flux_forward,
)
from astria.taylor_cache.forwards.dicache_forward import (
    dicache_forward,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

@maybe_allow_in_graph
class FluxSingleTransformerBlockTaylorCaching(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        hidden_states = taylorseer_flux_single_block_forward(
            self,
            hidden_states,
            temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlockTaylorCaching(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return taylorseer_flux_double_block_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


class FluxTransformer2DTaylorCachingModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    def __init__(
        self,
        transformer: "FluxTransformer2DModel",
        in_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = in_channels

        for block in transformer.transformer_blocks:
            block.__class__ = FluxTransformerBlockTaylorCaching
        for block in transformer.single_transformer_blocks:
            block.__class__ = FluxSingleTransformerBlockTaylorCaching

        self._config = {}

        self.add_module("pos_embed", transformer.pos_embed)
        self.add_module("time_text_embed", transformer.time_text_embed)
        self.add_module("context_embedder", transformer.context_embedder)
        self.add_module("x_embedder", transformer.x_embedder)
        self.add_module("transformer_blocks", transformer.transformer_blocks)
        self.add_module("single_transformer_blocks", transformer.single_transformer_blocks)
        self.add_module("norm_out", transformer.norm_out)
        self.add_module("proj_out", transformer.proj_out)

        self.pulid_ca = None
        self.config = transformer.config
        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(cls, tf: "FluxTransformer2DModel"):
        return cls(
            tf,
        )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        # validate if needed
        self._config = value

    def cache_context(self, *_args, **_kwargs):
        # returns a context manager that does nothing
        return nullcontext()

    def set_number_of_steps(self, steps: int):
        self.num_steps = steps

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        return taylorseer_flux_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_single_block_samples,
            return_dict,
            controlnet_blocks_repeat,
        )


class FluxTransformerBlockDiCacheCaching(nn.Module):
    """
    Block-level variant for DiCache. Mirrors structure of TaylorCaching but leaves
    forward delegation to normal block execution (DiCache logic is handled at model level).
    """
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # For DiCache we want plain block behavior, not Taylor caching hooks.
        # This mirrors the original FluxTransformerBlock forward:
        norm_hidden_states = self.norm1(hidden_states, temb)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states + self.ff(norm_hidden_states)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.ff_context(norm_encoder_hidden_states)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DDiCacheCachingModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    AttentionMixin,
):
    """
    Full model variant that wires in block subclasses and attaches dicache_forward
    as the forward method.
    """
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    def __init__(self, transformer: "FluxTransformer2DModel", in_channels: int = 64, **dicache_kwargs):
        super().__init__()
        self.out_channels = in_channels

        self._config = {}

        self.add_module("pos_embed", transformer.pos_embed)
        self.add_module("time_text_embed", transformer.time_text_embed)
        self.add_module("context_embedder", transformer.context_embedder)
        self.add_module("x_embedder", transformer.x_embedder)
        self.add_module("transformer_blocks", transformer.transformer_blocks)
        self.add_module("single_transformer_blocks", transformer.single_transformer_blocks)
        self.add_module("norm_out", transformer.norm_out)
        self.add_module("proj_out", transformer.proj_out)

        self.pulid_ca = None
        self.config = transformer.config
        self.gradient_checkpointing = False

        # dicache-specific parameters
        self.enable_dicache = True
        self.cnt = 0
        self.num_steps = dicache_kwargs.get("num_steps", 1)
        self.probe_depth = dicache_kwargs.get("probe_depth", 1)
        self.error_choice = dicache_kwargs.get("error_choice", "delta_y")
        self.rel_l1_thresh = dicache_kwargs.get("rel_l1_thresh", 0.1)
        self.rel_thresh_map = dicache_kwargs.get("rel_thresh_map", None)
        self.ret_ratio = dicache_kwargs.get("ret_ratio", 0.2)
        self.skip_end_ratio = dicache_kwargs.get("skip_end_ratio", 0.0)
        self.max_consec_skips = dicache_kwargs.get("max_consec_skips", 8)
        
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.residual_window = []
        self.probe_residual_window = []

        # If probe_depth wasn't specified in dicache_kwargs, enforce a safe default
        if "probe_depth" not in dicache_kwargs:
            self.probe_depth = 1  # recommend 1~5

        # attach dicache forward
        self.forward = types.MethodType(dicache_forward, self)

    @classmethod
    def from_transformer(cls, tf: "FluxTransformer2DModel", **kwargs):
        return cls(tf, **kwargs)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def clear_cache(self):
        # scalars / counters
        self.accumulated_rel_l1_distance = 0
        self.cnt = 0
        self.resume_flag = False
        self._consec_skips = 0

        # primary cached tensors
        self.previous_input = None
        self.previous_output = None
        self.previous_residual = None
        self.previous_probe_states = None
        self.previous_probe_residual = None
        self.previous_modulated_input = None  # if you actually use this elsewhere

        # windows
        self.residual_window = []
        self.probe_residual_window = []

    def cache_context(self, *_args, **_kwargs):
        return nullcontext()

    def set_number_of_steps(self, steps: int):
        self.num_steps = steps
