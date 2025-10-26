from fal_client import FalClient
from image_utils import load_image
from vertex_api import GeminiImageApi
from byteplus_client import ByteplusClient
from diffusers import ImagePipelineOutput

PARTNER_GEMINI_TUNE_ID = 3159068
PARTNER_REVE_TUNE_ID = 3279226
PARTNER_SEEDREAM_TUNE_ID = 3225353
PARTNER_RIVERFLOW_TUNE_ID = 3449218
PARTNER_RIVERFLOW_MINI_TUNE_ID = 3449219

class PartnerPipelineBase:
    def __call__(self, prompt, **kwargs):
        print("Using inferred Gemini images from prompt.images")
        images = [load_image(image) for image in prompt.images]
        return ImagePipelineOutput(images=images)

class GeminiPipeline(PartnerPipelineBase):

    def __call__(self, prompt, **kwargs):
        if prompt.images:
            return super().__call__(prompt, **kwargs)
        else:
            images = GeminiImageApi.gemini_edit_image(prompt)
            return ImagePipelineOutput(images=images)

class PartnerPipeline(PartnerPipelineBase):
    def __call__(self, prompt, **kwargs):
        if prompt.images:
            return super().__call__(prompt, **kwargs)
        else:
            if prompt.tune_id == PARTNER_SEEDREAM_TUNE_ID:
                images = ByteplusClient.seedream_edit(prompt)
            elif prompt.tune_id == PARTNER_REVE_TUNE_ID:
                images = FalClient.reve(prompt)
            else:
                raise ValueError("Invalid partner tune id")
            return ImagePipelineOutput(images=images)
