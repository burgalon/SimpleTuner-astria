# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import List

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel


@maybe_allow_in_graph
class RAG_FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def scale_lora_layers_according_to_region(self, lora_scalings: Dict[str, Any]):
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                for adapter_id, adapter_scale in lora_scalings.items():
                    module.set_scale(adapter_id, adapter_scale)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            SR_residual_list = []
            SR_norm_hidden_states_list = []
            SR_gate_list = []
            SR_mlp_hidden_states_list = []

            for SR_hidden_states, lora_scalings in zip(
                joint_attention_kwargs["SR_hidden_states_list"],
                joint_attention_kwargs["lora_regional_scaling"],
            ):
                self.scale_lora_layers_according_to_region(lora_scalings)
                SR_residual = SR_hidden_states
                SR_norm_hidden_states, SR_gate = self.norm(SR_hidden_states, emb=temb)
                SR_mlp_hidden_states = self.act_mlp(self.proj_mlp(SR_norm_hidden_states))
                SR_residual_list.append(SR_residual)
                SR_norm_hidden_states_list.append(SR_norm_hidden_states)
                SR_gate_list.append(SR_gate)
                SR_mlp_hidden_states_list.append(SR_mlp_hidden_states)
            joint_attention_kwargs["SR_norm_hidden_states_list"] = SR_norm_hidden_states_list

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            attn_output, SR_attn_output_list = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs
            )
        else:
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs
            )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            SR_hidden_states_list = []

            for SR_attn_output, SR_mlp_hidden_states, SR_gate, SR_residual, lora_scalings in zip(
                    SR_attn_output_list,
                    SR_mlp_hidden_states_list,
                    SR_gate_list,
                    SR_residual_list,
                    joint_attention_kwargs["lora_regional_scaling"],
                ):
                self.scale_lora_layers_according_to_region(lora_scalings)
                SR_hidden_states = torch.cat([SR_attn_output, SR_mlp_hidden_states], dim=2)
                SR_gate = SR_gate.unsqueeze(1)
                SR_hidden_states = SR_gate * self.proj_out(SR_hidden_states)
                SR_hidden_states = SR_residual + SR_hidden_states
                if SR_hidden_states.dtype == torch.float16:
                    SR_hidden_states = SR_hidden_states.clip(-65504, 65504)
                SR_hidden_states_list.append(SR_hidden_states)
            return hidden_states,SR_hidden_states_list

        return hidden_states


@maybe_allow_in_graph
class RAG_FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def scale_lora_layers_according_to_region(self, lora_scalings: Dict[str, Any]):
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                for adapter_id, adapter_scale in lora_scalings.items():
                    module.set_scale(adapter_id, adapter_scale)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            SR_norm_encoder_hidden_states_list = []
            SR_c_gate_msa_list = []
            SR_c_shift_mlp_list = []
            SR_c_scale_mlp_list = []
            SR_c_gate_mlp_list = []
            SR_encoder_hidden_states_list = joint_attention_kwargs["SR_encoder_hidden_states_list"]

            for SR_encoder_hidden_states, lora_scalings in zip(
                    SR_encoder_hidden_states_list,
                    joint_attention_kwargs["lora_regional_scaling"],
                ):
                self.scale_lora_layers_according_to_region(lora_scalings)
                SR_norm_encoder_hidden_states, SR_c_gate_msa, SR_c_shift_mlp, SR_c_scale_mlp, SR_c_gate_mlp = self.norm1_context(
                    SR_encoder_hidden_states, emb=temb
                )
                SR_norm_encoder_hidden_states_list.append(SR_norm_encoder_hidden_states)
                SR_c_gate_msa_list.append(SR_c_gate_msa)
                SR_c_shift_mlp_list.append(SR_c_shift_mlp)
                SR_c_scale_mlp_list.append(SR_c_scale_mlp)
                SR_c_gate_mlp_list.append(SR_c_gate_mlp)
            joint_attention_kwargs["SR_norm_encoder_hidden_states_list"] = SR_norm_encoder_hidden_states_list

        # Attention.
        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            attn_output, context_attn_output, SR_context_attn_output_list = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
        else:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            updated_SR_encoder_hidden_states_list = []

            for SR_context_attn_output, SR_c_gate_msa, SR_encoder_hidden_states, SR_c_scale_mlp, SR_c_shift_mlp, SR_c_gate_mlp, lora_scalings in zip(
                    SR_context_attn_output_list,
                    SR_c_gate_msa_list,
                    SR_encoder_hidden_states_list,
                    SR_c_scale_mlp_list,
                    SR_c_shift_mlp_list,
                    SR_c_gate_mlp_list,
                    joint_attention_kwargs["lora_regional_scaling"],
                ):
                self.scale_lora_layers_according_to_region(lora_scalings)
                SR_context_attn_output = SR_c_gate_msa.unsqueeze(1) * SR_context_attn_output
                SR_encoder_hidden_states = SR_encoder_hidden_states + SR_context_attn_output

                SR_norm_encoder_hidden_states = self.norm2_context(SR_encoder_hidden_states)
                SR_norm_encoder_hidden_states = SR_norm_encoder_hidden_states * (1 + SR_c_scale_mlp[:, None]) + SR_c_shift_mlp[:, None]

                SR_context_ff_output = self.ff_context(SR_norm_encoder_hidden_states)
                SR_encoder_hidden_states = SR_encoder_hidden_states + SR_c_gate_mlp.unsqueeze(1) * SR_context_ff_output
                if SR_encoder_hidden_states.dtype == torch.float16:
                    SR_encoder_hidden_states = SR_encoder_hidden_states.clip(-65504, 65504)
                updated_SR_encoder_hidden_states_list.append(SR_encoder_hidden_states)
            return encoder_hidden_states, hidden_states, updated_SR_encoder_hidden_states_list

        return encoder_hidden_states, hidden_states


class RAG_FluxTransformer2DModel(ModelMixin, PeftAdapterMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    # @register_to_config
    # def __init__(
    #     self,
    #     patch_size: int = 1,
    #     in_channels: int = 64,
    #     num_layers: int = 19,
    #     num_single_layers: int = 38,
    #     attention_head_dim: int = 128,
    #     num_attention_heads: int = 24,
    #     joint_attention_dim: int = 4096,
    #     pooled_projection_dim: int = 768,
    #     guidance_embeds: bool = False,
    #     axes_dims_rope: Tuple[int] = (16, 56, 56),
    # ):
    #     super().__init__()
    #     self.out_channels = in_channels
    #     self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

    #     self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

    #     text_time_guidance_cls = (
    #         CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
    #     )
    #     self.time_text_embed = text_time_guidance_cls(
    #         embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
    #     )

    #     self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
    #     self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

    #     self.transformer_blocks = nn.ModuleList(
    #         [
    #             FluxTransformerBlock(
    #                 dim=self.inner_dim,
    #                 num_attention_heads=self.config.num_attention_heads,
    #                 attention_head_dim=self.config.attention_head_dim,
    #             )
    #             for i in range(self.config.num_layers)
    #         ]
    #     )

    #     self.single_transformer_blocks = nn.ModuleList(
    #         [
    #             FluxSingleTransformerBlock(
    #                 dim=self.inner_dim,
    #                 num_attention_heads=self.config.num_attention_heads,
    #                 attention_head_dim=self.config.attention_head_dim,
    #             )
    #             for i in range(self.config.num_single_layers)
    #         ]
    #     )

    #     self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
    #     self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    #     self.gradient_checkpointing = False

    _supports_gradient_checkpointing = False
    _no_split_modules = ["RAG_FluxTransformerBlock", "RAG_FluxSingleTransformerBlock"]

    def __init__(
        self,
        transformer: "FluxTransformer2DModel",
        in_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = in_channels

        for block in transformer.transformer_blocks:
            block.__class__ = RAG_FluxTransformerBlock
        for block in transformer.single_transformer_blocks:
            block.__class__ = RAG_FluxSingleTransformerBlock

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
            # guidance_embeds=tf.config.guidance_embeds,
        )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def scale_lora_layers_according_to_region(self, lora_scalings: Dict[str, Any]):
        for module in self.modules():
            if isinstance(module, BaseTunerLayer):
                for adapter_id, adapter_scale in lora_scalings.items():
                    module.set_scale(adapter_id, adapter_scale)

    def HB_replace_hidden_states(self, hidden_states, HB_hidden_states_list_list, HB_m_offset_list,HB_n_offset_list,HB_m_scale_list,HB_n_scale_list, latent_h, latent_w, HB_idx):
        hidden_states = hidden_states.view(hidden_states.shape[0], latent_h,latent_w, hidden_states.shape[2])

        for HB_hidden_states_list, HB_m_offset, HB_n_offset, HB_m_scale, HB_n_scale in zip(HB_hidden_states_list_list, HB_m_offset_list, HB_n_offset_list, HB_m_scale_list, HB_n_scale_list):
            HB_hidden_states = HB_hidden_states_list[HB_idx]
            HB_hidden_states = HB_hidden_states.view(HB_hidden_states.shape[0], HB_n_scale,HB_m_scale, HB_hidden_states.shape[2])
            hidden_states[:, HB_n_offset:HB_n_offset+HB_n_scale, HB_m_offset:HB_m_offset+HB_m_scale, :] = HB_hidden_states

        hidden_states = hidden_states.view(hidden_states.shape[0], latent_h*latent_w, hidden_states.shape[3])
        HB_idx += 1

        return hidden_states, HB_idx
    
    def Repainting_replace_hidden_states(self, hidden_states, original_hidden_states_list, Repainting_HB_m_offset, Repainting_HB_n_offset, Repainting, latent_h, latent_w, Repainting_idx):
        original_hidden_states = original_hidden_states_list[Repainting_idx]
        original_hidden_states = original_hidden_states.view(original_hidden_states.shape[0], latent_h, latent_w, original_hidden_states.shape[2])
        hidden_states = hidden_states.view(hidden_states.shape[0], latent_h,latent_w, hidden_states.shape[2])
        original_hidden_states[:, Repainting_HB_n_offset:Repainting_HB_n_offset+Repainting.shape[1], Repainting_HB_m_offset:Repainting_HB_m_offset+Repainting.shape[2], :][Repainting == 1] = hidden_states[:, Repainting_HB_n_offset:Repainting_HB_n_offset+Repainting.shape[1], Repainting_HB_m_offset:Repainting_HB_m_offset+Repainting.shape[2], :][Repainting == 1]
        hidden_states = original_hidden_states.view(hidden_states.shape[0], latent_h*latent_w, hidden_states.shape[3])
        Repainting_idx += 1

        return hidden_states, Repainting_idx

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
        latent_h: int=None,
        latent_w: int=None,
        HB_hidden_states_list_list: List[List[torch.Tensor]] = None,
        HB_m_offset_list: List[int]=None,
        HB_n_offset_list: List[int]=None,
        HB_m_scale_list: List[int]=None,
        HB_n_scale_list: List[int]=None,
        return_hidden_states_list: bool = False,
        original_hidden_states_list: List[torch.Tensor] = None,
        Repainting_HB_m_offset: int=None,
        Repainting_HB_n_offset: int=None,
        Repainting: torch.Tensor = None,
        Repainting_single: int=False,
        lora_regional_scaling: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if return_hidden_states_list:
            hidden_states_list=[]
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        if HB_hidden_states_list_list is not None:
            HB_idx = 0
            hidden_states, HB_idx = self.HB_replace_hidden_states(
                hidden_states,
                HB_hidden_states_list_list,
                HB_m_offset_list,
                HB_n_offset_list,
                HB_m_scale_list,
                HB_n_scale_list,
                latent_h,
                latent_w,
                HB_idx,
            )

        if original_hidden_states_list is not None:
            Repainting_idx = 0
            hidden_states, Repainting_idx = self.Repainting_replace_hidden_states(
                hidden_states,
                original_hidden_states_list,
                Repainting_HB_m_offset,
                Repainting_HB_n_offset,
                Repainting,
                latent_h,
                latent_w,
                Repainting_idx,
            )

        if return_hidden_states_list:
            hidden_states_list.append(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            joint_attention_kwargs["SR_encoder_hidden_states_list"] = [
                self.context_embedder(SR_encoder_hidden_states) for SR_encoder_hidden_states in joint_attention_kwargs["SR_encoder_hidden_states_list"]
            ]
            joint_attention_kwargs['lora_regional_scaling'] = lora_regional_scaling

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
                    encoder_hidden_states, hidden_states, joint_attention_kwargs["SR_encoder_hidden_states_list"] = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if HB_hidden_states_list_list is not None:
                    hidden_states, HB_idx = self.HB_replace_hidden_states(
                        hidden_states,
                        HB_hidden_states_list_list,
                        HB_m_offset_list,
                        HB_n_offset_list,
                        HB_m_scale_list,
                        HB_n_scale_list,
                        latent_h,
                        latent_w,
                        HB_idx,
                    )

                if original_hidden_states_list is not None:
                    hidden_states, Repainting_idx = self.Repainting_replace_hidden_states(
                        hidden_states,
                        original_hidden_states_list,
                        Repainting_HB_m_offset,
                        Repainting_HB_n_offset,
                        Repainting,
                        latent_h,
                        latent_w,
                        Repainting_idx,
                    )

                if return_hidden_states_list:
                    hidden_states_list.append(hidden_states)

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
            joint_attention_kwargs["SR_hidden_states_list"] = [
                torch.cat([SR_encoder_hidden_states, hidden_states], dim=1)
                for SR_encoder_hidden_states in joint_attention_kwargs["SR_encoder_hidden_states_list"]
            ]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                if joint_attention_kwargs is not None and "SR_encoder_hidden_states_list" in joint_attention_kwargs:
                    hidden_states,joint_attention_kwargs["SR_hidden_states_list"] = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    ) 

                if HB_hidden_states_list_list is not None:
                    hidden_states_clone = hidden_states.clone()[:, encoder_hidden_states.shape[1] :, ...].view(hidden_states.shape[0], latent_h, latent_w, hidden_states.shape[2])

                    for HB_hidden_states_list, HB_m_offset, HB_n_offset, HB_m_scale, HB_n_scale in zip(
                            HB_hidden_states_list_list,
                            HB_m_offset_list,
                            HB_n_offset_list,
                            HB_m_scale_list,
                            HB_n_scale_list,
                        ):
                        HB_hidden_states = HB_hidden_states_list[HB_idx]
                        HB_hidden_states = HB_hidden_states[:, HB_hidden_states.shape[1]-HB_n_scale*HB_m_scale :, ...].view(HB_hidden_states.shape[0], HB_n_scale, HB_m_scale, HB_hidden_states.shape[2])
                        hidden_states_clone[:, HB_n_offset:HB_n_offset+HB_n_scale,HB_m_offset:HB_m_offset+HB_m_scale, :] = HB_hidden_states

                    hidden_states_clone = hidden_states_clone.view(hidden_states.shape[0], latent_h*latent_w, hidden_states.shape[2])
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = hidden_states_clone
                    HB_idx += 1

                if original_hidden_states_list is not None:
                    if Repainting_single:
                        hidden_states_clone = hidden_states.clone()[:, encoder_hidden_states.shape[1] :, ...].view(
                            hidden_states.shape[0], latent_h, latent_w, hidden_states.shape[2])
                        original_hidden_states = original_hidden_states_list[Repainting_idx]
                        original_hidden_states = original_hidden_states[:, encoder_hidden_states.shape[1] :, ...].view(
                            original_hidden_states.shape[0], latent_h, latent_w, original_hidden_states.shape[2])
                        original_hidden_states[:, Repainting_HB_n_offset:Repainting_HB_n_offset+Repainting.shape[1], Repainting_HB_m_offset:Repainting_HB_m_offset+Repainting.shape[2], :][Repainting == 1] = hidden_states_clone[:, Repainting_HB_n_offset:Repainting_HB_n_offset+Repainting.shape[1], Repainting_HB_m_offset:Repainting_HB_m_offset+Repainting.shape[2], :][Repainting == 1]
                        hidden_states_clone = original_hidden_states.view(hidden_states.shape[0], latent_h*latent_w, hidden_states.shape[2])
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = hidden_states_clone
                    Repainting_idx += 1

                if return_hidden_states_list:
                    hidden_states_list.append(hidden_states)

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)

        if HB_hidden_states_list_list is not None:
            hidden_states, HB_idx = self.HB_replace_hidden_states(
                hidden_states,
                HB_hidden_states_list_list,
                HB_m_offset_list,
                HB_n_offset_list,
                HB_m_scale_list,
                HB_n_scale_list,
                latent_h,
                latent_w,
                HB_idx,
            )

        if original_hidden_states_list is not None:
            hidden_states, Repainting_idx = self.Repainting_replace_hidden_states(
                hidden_states,
                original_hidden_states_list,
                Repainting_HB_m_offset,
                Repainting_HB_n_offset,
                Repainting,
                latent_h,
                latent_w,
                Repainting_idx,
            )

        if return_hidden_states_list:
            hidden_states_list.append(hidden_states)

        output = self.proj_out(hidden_states)

        if HB_hidden_states_list_list is not None:
            hidden_states, HB_idx = self.HB_replace_hidden_states(
                hidden_states,
                HB_hidden_states_list_list,
                HB_m_offset_list,
                HB_n_offset_list,
                HB_m_scale_list,
                HB_n_scale_list,
                latent_h,
                latent_w,
                HB_idx,
            )

        if original_hidden_states_list is not None:
            hidden_states, Repainting_idx = self.Repainting_replace_hidden_states(
                hidden_states,
                original_hidden_states_list,
                Repainting_HB_m_offset,
                Repainting_HB_n_offset,
                Repainting,
                latent_h,
                latent_w,
                Repainting_idx,
            )

        if return_hidden_states_list:
            hidden_states_list.append(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if return_hidden_states_list:
                    return (output,),hidden_states_list
            else:
                return (output,)

        return Transformer2DModelOutput(sample=output)