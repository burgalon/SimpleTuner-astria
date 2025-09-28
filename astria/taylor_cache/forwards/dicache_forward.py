from typing import Any, Dict, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_torch_version
from diffusers.utils import logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
import torch
import numpy as np

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def dicache_forward(
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
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Image-only caching version of dicache_forward:
    - All cache deltas (delta_x / delta_y), residuals, probe states, windows, etc.
      are computed **only over the image tokens** (hidden_states).
    - Text tokens (encoder_hidden_states) still flow through the network as usual,
      but are never cached or used to decide skipping/resuming.
    """

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # token/type embeddings
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

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # --------------------------- IMAGE-ONLY DICACHE ----------------------------
    if self.enable_dicache:
        # Online probe profiling (image-only deltas)
        if self.cnt <= int(self.ret_ratio * self.num_steps) or self.cnt == self.num_steps - 1:
            should_calc = True
            self.resume_flag = False
            self.accumulated_rel_l1_distance = 0
            # reset skip counter on any forced compute
            if not hasattr(self, "_consec_skips"):
                self._consec_skips = 0
            self._consec_skips = 0
        else:
            # Run probe blocks to get *probe image states* only
            test_hidden_states, test_encoder_hidden_states = hidden_states.clone(), encoder_hidden_states.clone()
            probe_blocks = self.transformer_blocks[0:self.probe_depth]
            for probe_block in probe_blocks:
                test_encoder_hidden_states, test_hidden_states = probe_block(
                    hidden_states=test_hidden_states,
                    encoder_hidden_states=test_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # IMAGE-ONLY deltas (no text concatenation)
            # previous_input / previous_probe_states are image-only tensors
            delta_x = (hidden_states - self.previous_input).abs().mean() / self.previous_input.abs().mean()
            delta_y = (test_hidden_states - self.previous_probe_states).abs().mean() / self.previous_probe_states.abs().mean()

            if self.error_choice == "delta_minus":
                error = (delta_y - delta_x).abs()
            elif self.error_choice == "delta_y":
                error = delta_y
            elif self.error_choice == "cosine":
                # Directional change only (scale-invariant)
                eps = 1e-8
                B = test_hidden_states.shape[0]
                cur = test_hidden_states.reshape(B, -1)
                prev = self.previous_probe_states.reshape(B, -1)
                cos = torch.nn.functional.cosine_similarity(cur, prev, dim=1, eps=eps)
                error = (1.0 - cos).mean()
            elif self.error_choice == "l2":
                # Relative L2 (norm ratio); optionally use RMS/RMSE flavor
                eps = 1e-8
                diff = (test_hidden_states - self.previous_probe_states)
                num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), ord=2, dim=1)
                den = torch.linalg.vector_norm(self.previous_probe_states.reshape(diff.shape[0], -1), ord=2, dim=1).clamp_min(eps)
                rel_l2 = (num / den).mean()
                error = rel_l2
            elif self.error_choice == "cosine_l1_hybrid":
                eps = 1e-8
                B = test_hidden_states.shape[0]
                cur = test_hidden_states.reshape(B, -1)
                prev = self.previous_probe_states.reshape(B, -1)
                cos = torch.nn.functional.cosine_similarity(cur, prev, dim=1, eps=eps)
                cos_dist = (1.0 - cos)  # [0, 2]
                error = torch.maximum(cos_dist.mean(), delta_y.detach())
            else:
                # Fallback to delta_y
                error = delta_y

            self.accumulated_rel_l1_distance += error

            # --------- pick current threshold from rel_thresh_map (or rel_l1_thresh) ---------
            # progress in [0,1]
            t, T = self.cnt, self.num_steps
            progress = t / max(T - 1, 1)

            # default to rel_l1_thresh
            current_thresh = self.rel_l1_thresh
            # optional rel_thresh_map:
            # can be a float or a list[{"start": float, "threshold": float}, ...]
            if hasattr(self, "rel_thresh_map") and self.rel_thresh_map is not None:
                if isinstance(self.rel_thresh_map, (int, float)):
                    current_thresh = float(self.rel_thresh_map)
                else:
                    # iterate regions in order of start; pick the last region whose start <= progress
                    # (cheap to sort every time; if you prefer, cache a sorted copy in __init__)
                    try:
                        regions = sorted(self.rel_thresh_map, key=lambda r: float(r.get("start", 0.0)))
                        for r in regions:
                            s = float(r.get("start", 0.0))
                            if progress >= s:
                                current_thresh = float(r.get("threshold", current_thresh))
                            else:
                                break
                    except Exception:
                        # if malformed, silently fall back to base threshold
                        current_thresh = self.rel_l1_thresh
            # -------------------------------------------------------------------------------

            if self.accumulated_rel_l1_distance < current_thresh:
                should_calc = False
                self.resume_flag = False
            else:
                should_calc = True
                self.resume_flag = True
                self.accumulated_rel_l1_distance = 0

            # ------- Cadence guard: cap consecutive skips -------
            if not hasattr(self, "_consec_skips"):
                self._consec_skips = 0

            if should_calc:
                # any compute resets the counter
                self._consec_skips = 0
            else:
                self._consec_skips += 1
                # if we've skipped too many times in a row, force a compute
                if self._consec_skips >= self.max_consec_skips:
                    should_calc = True
                    self.resume_flag = False
                    self._consec_skips = 0

        if not should_calc:
            # SKIP this step, synthesize image state from cached residuals (image-only)
            ori_hidden_states = hidden_states.clone()

            # Dynamic Cache Trajectory Alignment (image-only)
            if len(self.residual_window) >= 2:
                current_residual_indicator = (test_hidden_states - hidden_states)
                prev_img_minus2 = self.probe_residual_window[-2]
                prev_img_minus1 = self.probe_residual_window[-1]
                gamma = (
                    (current_residual_indicator - prev_img_minus2).abs().mean()
                    / (prev_img_minus1 - prev_img_minus2).abs().mean()
                ).clip(1, 1.5)
                hidden_states = hidden_states + self.residual_window[-2] + gamma * (self.residual_window[-1] - self.residual_window[-2])
            else:
                hidden_states = hidden_states + self.previous_residual

            # Update image-only caches for the next step
            self.previous_probe_states = test_hidden_states.clone()
            self.previous_input = ori_hidden_states.clone()

        else:
            # COMPUTE path (may resume from probe)
            ori_hidden_states = hidden_states.clone()  # image-only baseline

            if self.resume_flag:  # resume image & text from probe
                hidden_states = test_hidden_states
                encoder_hidden_states = test_encoder_hidden_states
                unpass_transformer_blocks = self.transformer_blocks[self.probe_depth:]
            else:
                unpass_transformer_blocks = self.transformer_blocks

            # Main transformer blocks
            for index_block, block in enumerate(unpass_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

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
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # ControlNet residual for block path
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    if controlnet_blocks_repeat:
                        hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

                # Record probe image features at the probe depth boundary
                if index_block == self.probe_depth - 1:
                    if self.cnt <= int(self.ret_ratio * self.num_steps) or self.cnt == self.num_steps - 1:
                        self.previous_probe_states = hidden_states.clone()
                    else:
                        # keep the *probe* image states we already computed
                        self.previous_probe_states = test_hidden_states.clone()

            # Single transformer blocks
            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

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
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # ControlNet residual for single-block path
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # NOTE: this retains your original indexing behavior
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            # Update image-only residual caches
            self.previous_residual = hidden_states - ori_hidden_states
            self.previous_probe_residual = self.previous_probe_states - ori_hidden_states
            self.previous_input = ori_hidden_states
            self.previous_output = hidden_states

            self.residual_window.append(self.previous_residual)
            self.probe_residual_window.append(self.previous_probe_residual)

    # ------------------------------ NO DICACHE ---------------------------------
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

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
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

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
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

    # ------------------------------- OUTPUT ------------------------------------
    # Now `hidden_states` is *image tokens only*. No need to slice away text.
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0
        self.residual_window = []
        self.probe_residual_window = []
        self._consec_skips = 0    # <-- reset cadence counter

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
