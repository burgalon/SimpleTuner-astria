import torch, os, logging
import random
from helpers.models.common import (
    VideoImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)
from diffusers import AutoencoderKLWan
from helpers.models.wan_t2i.transformer import WanTransformer3DModel
from helpers.models.wan_t2i.pipeline import WanPipeline

from helpers.training.tread import TREADRouter
from torch.nn import functional as F

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)


class Wan(VideoImageModelFoundation):
    NAME = "WanT2I"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "unipc"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_q", "to_k", "to_v", "proj"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = WanTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: WanPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "t2v-480p-1.3b-2.1"
    HUGGINGFACE_PATHS = {
        "t2v-480p-1.3b-2.1": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "t2v-480p-14b-2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        # "i2v-480p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        # "i2v-720p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
        },
    }

    def get_lora_target_layers(self):
        if self.config.lora_type.lower() == "standard":
            if self.config.wan_lora_target == "attention":
                logger.info('Selected WAN LoRA target attention')
                return [
                    "to_q",
                    "to_k",
                    "to_v",
                ]

            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(
                f"Unknown LoRA target type {self.config.lora_type}."
            )

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(
            81, self.config.validation_num_video_frames or 81
        )
        pipeline_kwargs["output_type"] = "pil"
        # replace embeds with prompt

        return pipeline_kwargs

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": masks,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "attention_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "negative_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def tread_init(self):
        """
        Initialize the TREAD model training method for Wan.
        """

        if (
            getattr(self.config, "tread_config", None) is None
            or getattr(self.config, "tread_config", None) is {}
            or getattr(self.config, "tread_config", {}).get("routes", None) is None
        ):
            logger.error(
                "TREAD training requires you to configure the routes in the TREAD config"
            )
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            self.config.tread_config["routes"],
        )

        logger.info("TREAD training is enabled for Wan")

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, masks = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(
                device=prompt_embeds.device
            ).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, masks

    def model_predict(self, prepared_batch):
        """
        Modify the existing model_predict to support TREAD with masked training.
        """
        wan_transformer_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(
                self.config.weight_dtype
            ),
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(
                self.config.weight_dtype
            ),
            "timestep": prepared_batch["timesteps"],
            "return_dict": False,
        }

        # For masking with TREAD, avoid dropping any tokens that are in the mask
        if (
            getattr(self.config, "tread_config", None) is not None
            and self.config.tread_config is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("conditioning_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                mask = prepared_batch["conditioning_pixel_values"]          # (B,C,H,W) or (B,C,T,H,W)
                mask = (mask.mean(1, keepdim=True) + 1) / 2                 # (B,1,[T,]H,W)

                latents = prepared_batch["latents"]
                is_video = latents.dim() == 5                                # True → video run

                if is_video:                                                 # ---------- video ----------
                    B, C, T, H, W = latents.shape
                    patch_t, patch_h, patch_w = (1, 2, 2)                    # or read from cfg
                    t_tokens, h_tokens, w_tokens = (
                        T // patch_t,
                        H // patch_h,
                        W // patch_w,
                    )

                    # ── NEW: ensure `mask` is 5-D ───────────────────────────────
                    if mask.dim() == 4:                                       # (B,1,H,W) → (B,1,T,H,W)
                        mask = mask.unsqueeze(2).expand(-1, -1, T, -1, -1)

                    mask_tok = F.interpolate(
                        mask,
                        size=(t_tokens, h_tokens, w_tokens),                  # 3 numbers ⇒ needs 5-D
                        mode="trilinear",
                        align_corners=False,
                    )
                    force_keep = mask_tok.squeeze(1).flatten(1) > 0.5         # (B, T*H*W)

                else:                                                         # ---------- image ----------
                    B, C, H, W = latents.shape
                    patch_h, patch_w = (2, 2)
                    h_tokens, w_tokens = H // patch_h, W // patch_w

                    mask_tok = F.interpolate(
                        mask,                                                 # 4-D input
                        size=(h_tokens, w_tokens),                            # 2 numbers ⇒ ok for 4-D
                        mode="bilinear",
                        align_corners=False,
                    )
                    force_keep = mask_tok.squeeze(1).flatten(1) > 0.5         # (B, H*W)

            wan_transformer_kwargs["force_keep_mask"] = force_keep.to(torch.bool)

        model_pred = self.model(**wan_transformer_kwargs)[0]

        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                f"{self.NAME} requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        if self.config.prediction_type is not None:
            logger.warning(
                f"{self.NAME} does not support prediction type {self.config.prediction_type}."
            )

        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 226 tokens, --tokenizer_max_length is ignored -!-"
            )
        self.config.tokenizer_max_length = 226
        if self.config.validation_num_inference_steps > 50:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} may be wasting compute with more than 50 steps. Consider reducing the value to save time."
            )
        if self.config.validation_num_inference_steps < 40:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} expects around 40 or more inference steps. Consider increasing --validation_num_inference_steps to 40."
            )
        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation to save on time.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 15

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str

    def loss(
        self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True
    ):
        """
        Computes the loss between the model prediction and the target.
        Optionally applies SNR weighting and a conditioning mask.
        """
        target = self.get_prediction_target(prepared_batch)
        model_pred = model_output["model_prediction"]
        if target is None:
            raise ValueError("Target is None. Cannot compute loss.")

        # Get loss type from config (default to l2 for backward compatibility)
        loss_type = getattr(self.config, "loss_type", "l2")

        if self.PREDICTION_TYPE == PredictionTypes.FLOW_MATCHING:
            # Flow matching always uses L2 loss
            loss = (model_pred.float() - target.float()) ** 2
        elif self.PREDICTION_TYPE in [
            PredictionTypes.EPSILON,
            PredictionTypes.V_PREDICTION,
        ]:
            # Check if we're using Huber or smooth L1 loss
            if loss_type in ["huber", "smooth_l1"]:
                # Get timesteps for the batch
                timesteps = prepared_batch["timesteps"]

                # For scheduled huber, we compute per-sample then average
                if getattr(self.config, "huber_schedule", "constant") != "constant":
                    batch_size = model_pred.shape[0]
                    losses = []

                    for i in range(batch_size):
                        # Get scheduled huber_c for this timestep
                        huber_c = self.compute_scheduled_huber_c(
                            timesteps[i : i + 1]
                        ).item()

                        # Compute loss for this sample
                        sample_loss = self.conditional_loss(
                            model_pred[i : i + 1].float(),
                            target[i : i + 1].float(),
                            reduction="none",
                            loss_type=loss_type,
                            huber_c=huber_c,
                        )
                        losses.append(sample_loss)

                    loss = torch.cat(losses, dim=0)
                else:
                    # Constant huber_c - can be computed all at once
                    huber_c = getattr(self.config, "huber_c", 0.1)
                    loss = self.conditional_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="none",
                        loss_type=loss_type,
                        huber_c=huber_c,
                    )

                # Apply SNR weighting if configured (for Huber/smooth L1)
                if self.config.snr_gamma is not None and self.config.snr_gamma > 0:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if (
                        self.noise_schedule.config.prediction_type
                        == PredictionTypes.V_PREDICTION.value
                    ):
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma
                                * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights

            else:
                if self.config.snr_gamma is None or self.config.snr_gamma == 0:
                    loss = self.config.snr_weight * F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                else:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if (
                        self.noise_schedule.config.prediction_type
                        == PredictionTypes.V_PREDICTION.value
                    ):
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma
                                * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights
        else:
            raise NotImplementedError(
                f"Loss calculation not implemented for prediction type {self.PREDICTION_TYPE}."
            )

        # Apply conditioning mask if needed
        conditioning_type = prepared_batch.get("conditioning_type")
        if conditioning_type == "mask" and apply_conditioning_mask:
            mask = prepared_batch["conditioning_pixel_values"].to(
                dtype=loss.dtype, device=loss.device
            )  # shape: (B, C_mask, H_src, W_src)

            B, T, C, H_out, W_out = loss.shape

            # 1) Expand mask across time
            mask = mask.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B, T, C_mask, H_src, W_src)

            # 2) Collapse C_mask→1 (choose or average)
            # Option A: pick first channel
            mask = mask[:, :, 0:1, :, :]                        # (B, T, 1, H_src, W_src)
            # Option B: average all channels
            # mask = mask.mean(dim=2, keepdim=True)

            # 3) Flatten batch+time for 2D interp
            mask = mask.reshape(B * T, 1, mask.shape[-2], mask.shape[-1])  # (B*T,1,H_src,W_src)

            # 4) Resize to match loss’s H_out, W_out
            mask = torch.nn.functional.interpolate(mask, size=(H_out, W_out), mode="area")   # still (B*T,1,H_out,W_out)

            # 5) Un-flatten back into (B, T, 1, H_out, W_out)
            mask = mask.view(B, T, 1, H_out, W_out)

            # 6) Normalize & apply
            mask = mask / 2 + 0.5
            loss = loss * mask
        elif conditioning_type == "segmentation" and apply_conditioning_mask:
            if random.random() < self.config.masked_loss_probability:
                mask_image = prepared_batch["conditioning_pixel_values"].to(
                    dtype=loss.dtype, device=loss.device
                )
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
                target_hw = loss.shape[-2], loss.shape[-1]  # == (64, 64)
                mask_image = torch.nn.functional.interpolate(
                    mask_image,
                    size=target_hw,
                    mode="area",
                )
                mask_image = mask_image / 2 + 0.5
                mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
                loss = loss * mask_image

        # Average over channels and spatial dims, then over batch.
        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()
        return loss