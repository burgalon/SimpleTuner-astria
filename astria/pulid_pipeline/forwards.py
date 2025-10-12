# pulid/pulid_dicache_forward.py
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils import logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_version

logger = logging.get_logger(__name__)

def _maybe_inject_pulid(self, hidden_states, ca_idx, layer_kind, layer_idx,
                        pul_id_embedding, pul_id_weight):
    """
    Applies a PulID CA module if the current layer index hits the interval.
    Returns (hidden_states, ca_idx).
    layer_kind: 'double' for transformer_blocks, 'single' for single_transformer_blocks
    layer_idx:  index within that kind (0-based within the kind, like the original code)
    """
    if self.pulid_ca is None or pul_id_embedding is None:
        return hidden_states, ca_idx

    if layer_kind == "double":
        cond = (layer_idx % self.pulid_ca.double_interval == 0)
    else:
        cond = (layer_idx % self.pulid_ca.single_interval == 0)

    if cond:
        hidden_states = hidden_states + pul_id_weight * self.pulid_ca.pulid_ca[ca_idx](
            pul_id_embedding, hidden_states
        )
        ca_idx += 1
    return hidden_states, ca_idx


def pulid_dicache_forward(
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
    pul_id_embedding: Optional[torch.Tensor] = None,
    pul_id_weight: Optional[float] = 1.0,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    assert self.pulid_ca is not None, "Call `set_pulid_ca(...)` before forward."

    # ---- LoRA scaling ----
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("`scale` in `joint_attention_kwargs` is ignored without PEFT backend.")

    # ---- embeddings ----
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    guidance = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning("3D txt_ids deprecated; pass 2D without batch dim.")
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning("3D img_ids deprecated; pass 2D without batch dim.")
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # --------------------------- IMAGE-ONLY DICACHE ----------------------------
    if self.enable_dicache:
        # Always compute the very early portion and the last step
        if self.cnt <= int(self.ret_ratio * self.num_steps) or self.cnt == self.num_steps - 1:
            should_calc = True
            self.resume_flag = False
            self.accumulated_rel_l1_distance = 0
            self._consec_skips = 0
        else:
            # ==== probe path over first `probe_depth` double blocks ====
            test_hidden_states = hidden_states.clone()
            test_encoder_hidden_states = encoder_hidden_states.clone()
            probe_ca_idx = 0  # PulID CA modules consumed in the probe

            for i, probe_block in enumerate(self.transformer_blocks[: self.probe_depth]):
                # run the block
                test_encoder_hidden_states, test_hidden_states = probe_block(
                    hidden_states=test_hidden_states,
                    encoder_hidden_states=test_encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                # PulID injection that *would* happen at this probe layer
                test_hidden_states, probe_ca_idx = _maybe_inject_pulid(
                    self, test_hidden_states, probe_ca_idx, "double", i,
                    pul_id_embedding, pul_id_weight
                )

            # Relative image-only deltas against cached probe/input
            delta_x = (hidden_states - self.previous_input).abs().mean() / self.previous_input.abs().mean()
            delta_y = (test_hidden_states - self.previous_probe_states).abs().mean() / self.previous_probe_states.abs().mean()

            # error metric selection
            choice = getattr(self, "error_choice", "delta_y")
            if choice == "delta_minus":
                error = (delta_y - delta_x).abs()
            elif choice == "delta_y":
                error = delta_y
            elif choice == "cosine":
                eps = 1e-8
                B = test_hidden_states.shape[0]
                cur = test_hidden_states.reshape(B, -1)
                prev = self.previous_probe_states.reshape(B, -1)
                cos = torch.nn.functional.cosine_similarity(cur, prev, dim=1, eps=eps)
                error = (1.0 - cos).mean()
            elif choice == "l2":
                eps = 1e-8
                diff = (test_hidden_states - self.previous_probe_states)
                num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), ord=2, dim=1)
                den = torch.linalg.vector_norm(self.previous_probe_states.reshape(diff.shape[0], -1), ord=2, dim=1).clamp_min(eps)
                error = (num / den).mean()
            elif choice == "cosine_l1_hybrid":
                eps = 1e-8
                B = test_hidden_states.shape[0]
                cur = test_hidden_states.reshape(B, -1)
                prev = self.previous_probe_states.reshape(B, -1)
                cos = torch.nn.functional.cosine_similarity(cur, prev, dim=1, eps=eps)
                cos_dist = (1.0 - cos)
                error = torch.maximum(cos_dist.mean(), delta_y.detach())
            else:
                error = delta_y

            self.accumulated_rel_l1_distance += error

            # threshold schedule
            t, T = self.cnt, self.num_steps
            progress = t / max(T - 1, 1)
            current_thresh = self.rel_l1_thresh
            if getattr(self, "rel_thresh_map", None) is not None:
                m = self.rel_thresh_map
                if isinstance(m, (int, float)):
                    current_thresh = float(m)
                else:
                    try:
                        regions = sorted(m, key=lambda r: float(r.get("start", 0.0)))
                        for r in regions:
                            if progress >= float(r.get("start", 0.0)):
                                current_thresh = float(r.get("threshold", current_thresh))
                            else:
                                break
                    except Exception:
                        current_thresh = self.rel_l1_thresh

            if self.accumulated_rel_l1_distance < current_thresh:
                should_calc = False
                self.resume_flag = False
            else:
                should_calc = True
                self.resume_flag = True
                self.accumulated_rel_l1_distance = 0

            # cadence guard
            if should_calc:
                self._consec_skips = 0
            else:
                self._consec_skips += 1
                if self._consec_skips >= self.max_consec_skips:
                    should_calc = True
                    self.resume_flag = False
                    self._consec_skips = 0

        if not should_calc:
            # ===== SKIP: synthesize image tokens from cache =====
            ori_hidden_states = hidden_states.clone()
            # test_hidden_states exists here because we only skip after probing
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

            # update image-only caches for next step
            self.previous_probe_states = test_hidden_states.clone()
            self.previous_input = ori_hidden_states.clone()

        else:
            # ===== COMPUTE: full pass, possibly resuming from probe =====
            ori_hidden_states = hidden_states.clone()

            ca_idx = 0
            # If we probed, continue from there; keep PulID CA index in sync
            if self.cnt > int(self.ret_ratio * self.num_steps) and self.cnt != self.num_steps - 1 and self.resume_flag:
                hidden_states = test_hidden_states
                encoder_hidden_states = test_encoder_hidden_states
                unpass_transformer_blocks = self.transformer_blocks[self.probe_depth:]
                ca_idx = probe_ca_idx
                start_double_idx = self.probe_depth  # global index across double blocks
            else:
                unpass_transformer_blocks = self.transformer_blocks
                start_double_idx = 0

            # We'll capture the state exactly at the probe boundary during this compute pass
            boundary_probe_state = None
            target_probe_idx = self.probe_depth - 1 if self.probe_depth > 0 else None

            # ---- double blocks ----
            for local_i, block in enumerate(unpass_transformer_blocks):
                if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states, encoder_hidden_states, temb, image_rotary_emb, **ckpt_kwargs
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # PulID injection uses index within *double* blocks
                global_double_idx = start_double_idx + local_i
                hidden_states, ca_idx = _maybe_inject_pulid(
                    self, hidden_states, ca_idx, "double", global_double_idx,
                    pul_id_embedding, pul_id_weight
                )

                # Record the probe-boundary features *after* PulID at the exact boundary
                if (not self.resume_flag) and (self.probe_depth > 0) and (global_double_idx == target_probe_idx):
                    boundary_probe_state = hidden_states.clone()

            # ---- single blocks ----
            for single_i, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False):
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states, encoder_hidden_states, temb, image_rotary_emb, **ckpt_kwargs
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # PulID single-interval condition uses index within single blocks
                hidden_states, ca_idx = _maybe_inject_pulid(
                    self, hidden_states, ca_idx, "single", single_i,
                    pul_id_embedding, pul_id_weight
                )

            # cache updated image residuals
            self.previous_residual = hidden_states - ori_hidden_states

            # set previous_probe_states at the *probe boundary*
            if self.resume_flag:
                # we *started* from the probe state computed earlier in this step
                self.previous_probe_states = test_hidden_states.clone()
            else:
                if self.probe_depth > 0 and boundary_probe_state is not None:
                    self.previous_probe_states = boundary_probe_state
                else:
                    # probe_depth == 0 (probe state is "before any block")
                    self.previous_probe_states = ori_hidden_states.clone()

            self.previous_probe_residual = self.previous_probe_states - ori_hidden_states
            self.previous_input = ori_hidden_states
            self.previous_output = hidden_states

            self.residual_window.append(self.previous_residual)
            self.probe_residual_window.append(self.previous_probe_residual)

    # ------------------------------- OUTPUT ------------------------------------
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0
        self.residual_window = []
        self.probe_residual_window = []
        self._consec_skips = 0

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
