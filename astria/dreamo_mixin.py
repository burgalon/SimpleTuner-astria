import re

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.transforms.functional import normalize

from astria_utils import HUMAN_CLASS_NAMES
from image_utils import load_image

from dreamo_pipeline.pipeline import DreamOPipeline
from dreamo_pipeline.util import (
    img2tensor,
    resize_numpy_image_area,
    resize_numpy_image_long,
    tensor2img,
)

from pulid_pipeline.pulid_ext import PULID_LOCAL_DIR, PuLID

from vton_mixin import (
    VTON_CATEGORIES,
)


REF_RES = 1024


class DreamOMixin:
    @torch.no_grad()
    def get_dreamo_align_face(self, img):
        # the face preprocessing code is same as PuLID
        self.dreamo_face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.dreamo_face_helper.read_image(image_bgr)
        self.dreamo_face_helper.get_face_landmarks_5(only_center_face=True)
        self.dreamo_face_helper.align_warp_face()
        if len(self.dreamo_face_helper.cropped_faces) == 0:
            return None
        align_face = self.dreamo_face_helper.cropped_faces[0]

        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(torch.device("cuda"))
        parsing_out = self.dreamo_face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, input)
        face_features_image = tensor2img(face_features_image, rgb2bgr=False)

        return face_features_image
    
    def prep_dream_kwargs(self, prompt, kwargs):
        ref_conds = []
        debug_images = []
        for idx, match_groups in enumerate(re.findall(self.reference_pattern_re, prompt.text)):
            _type, token, scale = match_groups
            tune = next(iter([tune for tune in prompt.tunes if tune.token == token or str(tune.id) == token]), None)
            if not tune:
                raise Exception(f"Token {token} not found in prompt {prompt.id} tokens={prompt.tunes}")
            if not tune.face_swap_images:
                raise Exception(f"Token {token} has no face_swap_images")
            
            ref_task = "ip"

            # Clean the match from the prompt
            # also needs to be cleaned for VTON
            prompt.text = prompt.text.replace(f"<{_type}:{token}:{scale}>", "")

            if tune.name == "style":
                ref_task = "style"
            if tune.name in HUMAN_CLASS_NAMES:
                ref_task = "id"

            ref_image = load_image(tune.face_swap_images[0])
            if ref_task == "id":
                ref_image = resize_numpy_image_long(np.array(ref_image), 1024)
                ref_image = self.get_dreamo_align_face(ref_image)
            elif ref_task != "style":
                ref_image = self.remove_background([ref_image])[0]
                # create a white background the same size
                white_bg = Image.new("RGBA", ref_image.size, (255, 255, 255, 255))
                # composite your (now-RGBA) image over the white
                ref_image = Image.alpha_composite(white_bg, ref_image).convert("RGB")
            if ref_task != "id":
                ref_image = resize_numpy_image_area(np.array(ref_image), REF_RES * REF_RES)
            debug_images.append(ref_image)
            ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    'img': ref_image,
                    'task': ref_task,
                    'idx': idx + 1,
                }
            )

        if len(ref_conds) == 0:
            raise Exception('No references could be found')

        kwargs['ref_conds'] = ref_conds

    def init_dreamo(self, pipeline, device):
        PuLID.download_models()
        self.dreamo_face_helper = PuLID.init_face_helper(device, PULID_LOCAL_DIR)
        return DreamOPipeline(
            scheduler=pipeline.scheduler,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer_2=pipeline.tokenizer_2,
            vae=pipeline.vae,
            transformer=pipeline.transformer,
            use_turbo=False,
        ).to(device)

    def cleanup_dreamo(self, pipe):
        # This is a special method on the dreamo pipeline
        pipe.reset_t5()

        # Drop dreamo refs
        self.dreamo_face_helper = None
