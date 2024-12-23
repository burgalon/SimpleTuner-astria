import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from insightface.app import FaceAnalysis
from ultralytics import YOLO

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from face_cropping import (
    YOLO_FACE_MODEL,
    Bbox,
    filter_by_ratio,
    filter_k_largest, 
    ultralytics_predict,
)
from pulid_ext import PuLID
from transformer_multi import FluxTransformer2DModelWithPulIDMultiPerson


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def get_face_bboxes(
    image: Image.Image,
    yolo: YOLO,
    device: torch.device=None,
    BASE_TRAIN_RESOLUTION: int=1024,
) -> tuple[list[Bbox], list[Bbox]]:
    pred = ultralytics_predict(
        yolo,
        image=image,
        confidence=0.3,
        device=device,
        classes=None,  # ad_model_classes,
    )
    pred = filter_by_ratio(pred, low=0.003, high=1)
    # Remove or modify this if you *do* want to filter down to K largest bboxes:
    pred = filter_k_largest(pred, k=0)
    
    # If no bounding boxes found, return empty lists
    if len(pred.bboxes) == 0:
        return [], []
    
    # Save the original bounding boxes for reference
    pred.orig_bboxes = pred.bboxes

    # Enlarge all bounding boxes for face+hair+neck+ears, etc.
    enlarged_bboxes = []
    for (x1, y1, x2, y2) in pred.bboxes:
        w = x2 - x1
        h = y2 - y1
        new_x1 = x1 - 0.1 * w
        new_y1 = y1 - 0.1 * h
        new_x2 = x2 + 0.1 * w
        new_y2 = y2 + 0.1 * h
        enlarged_bboxes.append([new_x1, new_y1, new_x2, new_y2])

    # Apply the bounding box expansions to meet BASE_TRAIN_RESOLUTION
    final_bboxes = []
    for i, bbox in enumerate(enlarged_bboxes):
        orig_bbox = pred.orig_bboxes[i]
        (x1, y1, x2, y2) = bbox
        
        # Enforce minimum width
        w = x2 - x1
        if w < BASE_TRAIN_RESOLUTION:
            diff = BASE_TRAIN_RESOLUTION - w
            x1 -= diff // 2
            x2 += diff // 2
        
        # Enforce minimum height
        h = y2 - y1
        if h < BASE_TRAIN_RESOLUTION:
            diff = BASE_TRAIN_RESOLUTION - h
            y1 -= diff // 2
            y2 += diff // 2

        final_bboxes.append((x1, y1, x2, y2))

    # Return the final list of bboxes and the original list if needed
    return final_bboxes, pred.orig_bboxes


def crop_faces_square_resize(
    image: Image.Image,
    face_bboxes: list[tuple[float, float, float, float]],
    output_size: int = 640,
) -> list[Image.Image]:
    """
    Takes an image and a list of bounding boxes (x1, y1, x2, y2) for faces,
    then returns a list of face-cropped PIL images, each resized to output_size x output_size.
    """
    face_crops = []
    img_width, img_height = image.size

    for (x1, y1, x2, y2) in face_bboxes:
        # 1. Convert float coords to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp the coordinates to remain within image boundaries
        x1 = max(0, min(img_width, x1))
        y1 = max(0, min(img_height, y1))
        x2 = max(0, min(img_width, x2))
        y2 = max(0, min(img_height, y2))

        # 2. Determine the largest side (to force a square)
        w = x2 - x1
        h = y2 - y1
        side = max(w, h)

        # 3. Center the square around the midpoint of the bbox
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        half_side = side // 2

        left = mid_x - half_side
        top = mid_y - half_side
        right = left + side
        bottom = top + side

        # Clamp again so we don't exceed image boundaries
        if left < 0:
            left = 0
            right = side
        if top < 0:
            top = 0
            bottom = side
        if right > img_width:
            right = img_width
            left = img_width - side
        if bottom > img_height:
            bottom = img_height
            top = img_height - side

        # 4. Crop the image to a square
        face_crop = image.crop((left, top, right, bottom))

        # 5. Resize to the desired output_size x output_size
        face_crop = face_crop.resize((output_size, output_size), resample=Image.LANCZOS)

        face_crops.append(face_crop)

    return face_crops


def crop_faces_square_resize_with_padding(
    image: Image.Image,
    face_bboxes: list[tuple[float, float, float, float]],
    output_size: int = 640,
    expansion_factor: float = 1.2
) -> list[Image.Image]:
    """
    Takes an image and a list of bounding boxes (x1, y1, x2, y2) for faces,
    expands the bounding box, pads it to form a square, and resizes it to output_size x output_size.
    """
    face_crops = []
    img_width, img_height = image.size

    for (x1, y1, x2, y2) in face_bboxes:
        # Convert float coords to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp the coordinates to remain within image boundaries
        x1 = max(0, min(img_width, x1))
        y1 = max(0, min(img_height, y1))
        x2 = max(0, min(img_width, x2))
        y2 = max(0, min(img_height, y2))
        
        # Expand the bounding box by the given factor
        w = x2 - x1
        h = y2 - y1
        side = max(w, h)
        expansion = int(side * (expansion_factor - 1) / 2)
        
        x1 -= expansion
        y1 -= expansion
        x2 += expansion
        y2 += expansion

        # Clamp expanded coordinates to remain within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        # Determine the square bounding box
        new_w = x2 - x1
        new_h = y2 - y1
        square_side = max(new_w, new_h)

        left = x1 - (square_side - new_w) // 2
        top = y1 - (square_side - new_h) // 2
        right = left + square_side
        bottom = top + square_side

        # Clamp the square coordinates to remain within image boundaries
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, left + square_side)
        bottom = min(img_height, top + square_side)

        # Crop the image
        face_crop = image.crop((left, top, right, bottom))

        # Pad the image to ensure it's square
        pad_width = max(0, square_side - (right - left))
        pad_height = max(0, square_side - (bottom - top))
        padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
        face_crop = ImageOps.expand(face_crop, border=padding, fill=(255, 255, 255))  # Fill with white

        # Resize to the desired output size
        face_crop = face_crop.resize((output_size, output_size), resample=Image.LANCZOS)

        face_crops.append(face_crop)

    return face_crops


class FaceModelEvaluator():
    def __init__(self, pretrained_model_name_or_path='buffalo_l', **kwargs):
        """
        kwargs["baseline_images"] should be a list of lists of PIL Images.
        Each sub-list in baseline_images corresponds to a single person
        with multiple images. We average the face embeddings within each sub-list
        to get a single embedding per person.
        """
        self.app = FaceAnalysis(
            name=pretrained_model_name_or_path or 'buffalo_l',
            root='/home/user/storage/faceanalysis',  # ./faceanalysis
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # baseline_images: list[list[Image.Image]]
        baseline_images = kwargs["baseline_images"]

        self.people_embeddings = []  # Will hold 1 averaged embedding per person

        # For each person’s images:
        for person_images in tqdm.tqdm(baseline_images, desc="Generating baseline face embeds..."):
            person_embeds = []

            # For each Image in that person's sub-list
            for pil_img in person_images:
                # Convert PIL to BGR NumPy for face detection
                img_np = np.array(pil_img)  # shape: (H, W, 3), in RGB
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                faces = self.app.get(img_bgr)
                if not faces:
                    # No face found in this image; skip it
                    continue

                # We'll just take the first face's embedding (assuming 1 face per image)
                try:
                    emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)  # shape (1, embed_dim)
                    person_embeds.append(emb)
                except IndexError:
                    pass

            
            if len(person_embeds) == 0:
                raise ValueError("Unable to extract any faces from these images")
            else:
                # Average across all images for that person
                # person_embeds is a list of shape (1, embed_dim) Tensors
                # Stack them => (num_images, embed_dim), then .mean(dim=0) => (embed_dim,)
                avg_emb = torch.stack(person_embeds, dim=0).mean(dim=0, keepdim=True).squeeze(0)
                # shape => (1, embed_dim), to keep consistent shape of [1, embed_dim]
                self.people_embeddings.append(avg_emb)

    def get_embedding(self, img: Image.Image) -> torch.Tensor:
        """
        Given a PIL image (with presumably one face), returns that face's embedding.
        If no face is found, returns None.
        """
        img_np = np.array(img)  
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces = self.app.get(img_bgr)
        if not faces:
            return None

        return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    def get_face_embedding_with_padding(self, img, max_retries=10, padding_increment=16):
        """
        Tries to get a face embedding from the model. If no embedding is returned,
        the image is padded with white pixels and retried up to max_retries times.

        Parameters:
            img (PIL.Image.Image): Input image.
            max_retries (int): Maximum number of retries with additional padding.
            padding_increment (int): Number of pixels to pad on each side per retry.

        Returns:
            np.ndarray: Face embedding if successful.
            
        Raises:
            ValueError: If no embedding is found after max_retries.
        """
        for attempt in range(max_retries):
            face_emb = self.get_embedding(img)  # Call your model to get the embedding
            if face_emb is not None:
                return face_emb

            # Log or print retry attempt
            print(f"Retry {attempt + 1}: Padding the image with {padding_increment} pixels.")

            # Pad the image with white pixels
            img = ImageOps.expand(img, border=padding_increment, fill=(255, 255, 255))

        # If no embedding is found after max_retries
        raise ValueError("Failed to get face embedding after 10 retries with padding.")

    def find_face_image_pairs(
        self,
        images: list[Image.Image],
        bboxes: list[Bbox],
    ) -> list[tuple[int, Bbox]]:
        """
        Using a list of images and self.people_embeddings, return a list of
        [int, Bbox] where the int refers to the index of self.people_embeddings
        that the bbox corresponds to. Ensures that indexes are unique across matches.
        
        If no face is found or all embeddings are None, -1 is used as a fallback index.

        Returns:
            A list of length len(images), where each element is:
                (unique_best_person_index, corresponding_bbox)
        """
        if len(images) != len(bboxes):
            raise ValueError(
                "Mismatch: images and bboxes must have the same length. "
                f"Got {len(images)} vs {len(bboxes)}."
            )

        results = []
        assigned_indexes = set()  # Track which indexes have been assigned

        for img, bbox in zip(images, bboxes):
            # 1. Extract embedding for the current face image
            face_emb = self.get_face_embedding_with_padding(img)  # shape: (1, embed_dim) or None if no face

            if face_emb is None:
                # No face or embedding was found in this image
                results.append((-1, bbox))
                continue

            # 2. Find the best match among self.people_embeddings
            best_index = -1
            best_score = float("-inf")

            for i, person_emb in enumerate(self.people_embeddings):
                # Skip already assigned indexes
                if i in assigned_indexes:
                    continue

                # Compute cosine similarity
                sim_score = F.cosine_similarity(person_emb, face_emb).item()

                if sim_score > best_score:
                    best_score = sim_score
                    best_index = i

            # 3. Append the pair (best_index, Bbox) if a valid match is found
            if best_index != -1:
                assigned_indexes.add(best_index)
                results.append((best_index, bbox))
            else:
                # No match found for this face
                results.append((-1, bbox))

        return results


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxPipelineMultiPersonWithPulID(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        pulid: PuLID,
    ):
        super().__init__()
        transformer = FluxTransformer2DModelWithPulIDMultiPerson.from_transformer(transformer)
        transformer.set_pulid_ca(pulid.model)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            pulid=pulid,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def _get_pulid_embeddings(
        self,
        images: Optional[List[Union[Image.Image, np.ndarray]]],
        device: Optional[torch.device] = torch.device('cuda'),
        # cal_uncond=False, TODO
    ) -> torch.Tensor:
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        embeds = []
        for image in images:
            embeds_single, _ = self.pulid.get_id_embedding(
                image,
                cal_uncond=False,
                device=device,
                dtype=dtype,
            )
            if embeds is not None:
                embeds.append(embeds_single.to(dtype=dtype))
            break
        

        embed_average = torch.stack(embeds, dim=0).mean(0)
        return embed_average

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = torch.Generator(),
        generator_seed: int=12345,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        id_images: Optional[
            list[list[Union[Image.Image, np.ndarray]]]
        ] = None,
        id_image_scales: Optional[list[float]] = None,
        pulid_skip_timesteps: int=0,
        pulid_skip_end_timesteps: int=-1,
        # id_negative_image_embeddings: Optional[torch.Tensor] = None, TODO
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            id_image: PIL or ndarray image for PulID embeddings.
            id_image_embeddings: Embeddings for PulID computed ahead of time.
            id_image_scale: Amount to scale the application of PulID embedding.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if num_images_per_prompt > 1:
            raise ValueError('This pipeline currently only handles 1 image per prompt')

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if pulid_skip_end_timesteps == -1:
            pulid_skip_end_timesteps = num_inference_steps

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if id_images is None:
            raise ValueError("id_images is required")
        
        if id_image_scales is not None and len(id_image_scales) != len(id_images):
            raise ValueError("length of id_images must be equivalent to id_image_scales")
            
        pul_id_embeddings = []
        for id_image_list in id_images:
            pul_id_embeddings.append(self._get_pulid_embeddings(
                id_image_list,
                device=device,
            ))
        if id_image_scales is None:
            id_image_scales = [1.0 for _ in range(len(id_images))]

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_initial = latents  # Preserve the original latents

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6.1 Denoising loop for GT image, without facial injection
        generator.manual_seed(generator_seed)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Post-process image with segmentation, zero-shot the assignment of
        # the closest faces using cosine similarity.
        latents = latents_initial
        yolo = YOLO(YOLO_FACE_MODEL)
        face_bboxes = get_face_bboxes(image[0], yolo, device=device)[1]
        print('face bboxes', face_bboxes)
        image[0].save('original_image.png')
        face_images = crop_faces_square_resize_with_padding(image[0], face_bboxes)
        for idx, img in enumerate(face_images):
            img.save(f'face_image_{idx}.png')
        face_model = FaceModelEvaluator(baseline_images=id_images)
        index_bbox_pairs = face_model.find_face_image_pairs(face_images, face_bboxes)

        face_id_embedding_bbox_pairs = [
            (pul_id_embeddings[idx], bbox)
            for idx, bbox in index_bbox_pairs
        ]

        # 6.2 Denoising loop for GT image, facial injection
        # import debugpy
        # debugpy.listen(('0.0.0.0', 11566))
        # debugpy.wait_for_client()
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        generator.manual_seed(generator_seed)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                pul_id_embeddings_bounding_boxes = None
                pul_id_weights = None
                if i >= pulid_skip_timesteps and i < pulid_skip_end_timesteps:
                    pul_id_embeddings_bounding_boxes = face_id_embedding_bbox_pairs
                    pul_id_weights = id_image_scales

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    pul_id_embeddings_bounding_boxes=pul_id_embeddings_bounding_boxes,
                    pul_id_weights=pul_id_weights,
                    pixel_height=height,
                    pixel_width=width,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
