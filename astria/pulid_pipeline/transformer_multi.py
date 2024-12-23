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


from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from face_cropping import Bbox

from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedFluxAttnProcessor2_0,
)

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
)
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_utils import ModelMixin

from pulid_ext import PuLModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_crop_masks(
    final_bboxes: list[tuple[float, float, float, float]],
    full_height: int,
    full_width: int,
    device: torch.device = None,
    flatten_for_sequence: bool = True,
) -> list[torch.Tensor]:
    """
    For each bounding box (x1, y1, x2, y2):
      1) Create a pixel mask of size (1, full_height, full_width).
      2) Downsample to (1, out_height, out_width).
      3) Optionally flatten to (out_height * out_width, [cond_dim]) if needed
    """
    mask_images = []
    out_height = full_height // 16
    out_width = full_width // 16
    itr = 0
    for (x1, y1, x2, y2) in final_bboxes:
        # Round and clip
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        x1, x2 = max(0, x1), min(full_width, x2)
        y1, y2 = max(0, y1), min(full_height, y2)

        # 1) Create full-res pixel mask
        mask_image = torch.zeros(
            (1, full_height, full_width),
            dtype=torch.float32,
            device=device
        )
        mask_image[:, y1:y2, x1:x2] = 1.0

        from PIL import Image
        mask_cpu = mask_image.squeeze(0).cpu()
        mask_cpu = (mask_cpu * 255).byte()  # Convert from [0,1] to [0,255]
        pil_mask = Image.fromarray(mask_cpu.numpy(), mode="L")  
        pil_mask.save(f"mask_fullres_{itr}.png")

        # 2) Downsample to match the model’s patch/latent shape
        #    shape becomes (1, out_height, out_width)
        mask_image = nn.functional.interpolate(
            mask_image.unsqueeze(0),  # shape (B=1, C=1, H, W)
            size=(out_height, out_width),
            mode="nearest-exact"
        ).squeeze(0)  # back to (1, out_height, out_width)

        if flatten_for_sequence:
            # 3A) Flatten from (1, H', W') -> (H'*W', 1)
            mask_image = mask_image.view(1, -1).transpose(0, 1).unsqueeze(0)
            # shape is now (H'*W', 1)

        itr += 1

        mask_images.append(mask_image)

    return mask_images


# ModelMixin, PeftAdapterMixin, FromOriginalModelMixin, ConfigMixin, ModelMixin
class FluxTransformer2DModelWithPulIDMultiPerson(ModelMixin):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    def __init__(
        self,
        transformer: FluxTransformer2DModel,
        in_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = in_channels

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

    @classmethod
    def from_transformer(cls, tf: FluxTransformer2DModel):
        return cls(
            tf,
            # guidance_embeds=tf.config.guidance_embeds,
        )

    def set_pulid_ca(self, pulid_ca: PuLModel):
        """
        This is not registered as a module, so will not be affected by .to(...)
        """
        self.pulid_ca = pulid_ca

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
        pul_id_embeddings_bounding_boxes: Optional[list[tuple[torch.Tensor], Bbox]] = None,
        pul_id_weights: Optional[list[float]] = 1.0,
        pixel_height: Optional[int]=None,
        pixel_width: Optional[int]=None,
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
            pul_id_embedding: Tensor for the PulID model embedding, optional.
            pul_id_weight: weight to apply the pul_id_embedding.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        assert self.pulid_ca is not None, 'self.pulid_id must be set, please set it using the `set_pulid_ca` method'

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

        ca_idx = 0
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if (
                pul_id_embeddings_bounding_boxes is not None
                and index_block % self.pulid_ca.double_interval == 0
            ):
                masks = get_crop_masks(
                    [pair[1] for pair in pul_id_embeddings_bounding_boxes],
                    pixel_height,
                    pixel_width,
                    hidden_states.device,
                )
                embeds_and_masks = [
                    (pul_id_embeddings_bounding_boxes[idx][0], masks[idx])
                    for idx in range(len(pul_id_embeddings_bounding_boxes))
                ]
                for idx, id_data in enumerate(embeds_and_masks):
                    pulid_embed, mask = id_data
                    pulid_layer_output = self.pulid_ca.pulid_ca[ca_idx](
                        pulid_embed,
                        hidden_states,
                    )
                    pulid_layer_output *= mask
                    
                    hidden_states = hidden_states + pul_id_weights[idx] * pulid_layer_output
                ca_idx += 1

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

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if (
                pul_id_embeddings_bounding_boxes is not None
                and index_block % self.pulid_ca.single_interval == 0
            ):
                img, txt = hidden_states[
                    :, encoder_hidden_states.shape[1]:, ...
                ], hidden_states[
                    :, :encoder_hidden_states.shape[1], ...
                ]

                masks = get_crop_masks(
                    [pair[1] for pair in pul_id_embeddings_bounding_boxes],
                    pixel_height,
                    pixel_width,
                    hidden_states.device,
                )
                embeds_and_masks = [
                    (pul_id_embeddings_bounding_boxes[idx][0], masks[idx])
                    for idx in range(len(pul_id_embeddings_bounding_boxes))
                ]
                for idx, id_data in enumerate(embeds_and_masks):
                    pulid_embed, mask = id_data
                    pulid_layer_output = self.pulid_ca.pulid_ca[ca_idx](
                        pulid_embed,
                        img,
                    )
                    pulid_layer_output *= mask

                    img = img + pul_id_weights[idx] * self.pulid_ca.pulid_ca[ca_idx](
                        pulid_embed,
                        img,
                    )
                ca_idx += 1
                hidden_states = torch.cat((txt, img), 1)

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
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
