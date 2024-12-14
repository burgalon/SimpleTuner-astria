from functools import partial
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms
import torch, logging, os
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
from helpers.training.state_tracker import StateTracker
import torch.nn.functional as F
import tqdm

logger = logging.getLogger("ModelEvaluator")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

model_evaluator_map = {
    "clip": "CLIPModelEvaluator",
    "face": "FaceModelEvaluator",
}

class ModelEvaluator:
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        raise NotImplementedError("Subclasses is incomplete, no __init__ method was found.")

    def evaluate(self, images, prompts, **kwargs):
        raise NotImplementedError("Subclasses should implement the evaluate() method.")

    @staticmethod
    def from_config(args, **kwargs):
        """Instantiate a ModelEvaluator from the training config, if set to do so."""
        if not StateTracker.get_accelerator().is_main_process:
            return None
        if args.evaluation_type is not None and args.evaluation_type.lower() != "" and args.evaluation_type.lower() != "none":
            model_evaluator = model_evaluator_map[args.evaluation_type]
            return globals()[model_evaluator](args.pretrained_evaluation_model_name_or_path, **kwargs)

        return None


class CLIPModelEvaluator(ModelEvaluator):
    def __init__(self, pretrained_model_name_or_path='openai/clip-vit-large-patch14-336'):
        self.clip_score_fn = partial(clip_score,
            model_name_or_path=pretrained_model_name_or_path or 'openai/clip-vit-large-patch14-336')
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

    def evaluate(self, images, prompts, **kwargs):
        # Preprocess images
        images_tensor = torch.stack([self.preprocess(img) * 255 for img in images])
        # Compute CLIP scores
        result = self.clip_score_fn(images_tensor, prompts).detach().cpu()

        return result


class FaceModelEvaluator(ModelEvaluator):
    def __init__(self, pretrained_model_name_or_path='buffalo_l', **kwargs):
        self.app = FaceAnalysis(
            name=pretrained_model_name_or_path or 'buffalo_l',
            root='./faceanalysis',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        baseline_images = kwargs["baseline_images"]
        embeds = []
        for baseline_image in tqdm.tqdm(baseline_images, desc="Generating baseline face embeds..."):
            image = cv2.imread(baseline_image)
            faces = self.app.get(image)

            # Ignore missing faces
            try:
                faceid_embed_gt = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            except IndexError:
                continue
            embeds.append(faceid_embed_gt)

        self.faceid_embed_gt = torch.stack(embeds, dim=0).mean(0)

    def evaluate(self, images, prompts, **kwargs):
        assert len(images) == 1, 'too many images passed to FaceModelEvaluator'
        experimental_img_bgr = None
        if isinstance(images[0], Image.Image):
            experimental_img = np.array(images[0])  # now shape: (H, W, 3), RGB
            experimental_img_bgr = cv2.cvtColor(experimental_img, cv2.COLOR_RGB2BGR)
        elif isinstance(images[0], np.ndarray):
            # If it's already a NumPy array, we must ensure it's in BGR for FaceAnalysis
            # E.g., if it is RGB, we convert:
            # experimental_img_bgr = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
            # Otherwise, if it's already BGR, skip the conversion.
            experimental_img_bgr = images[0]
        else:
            raise ValueError("Unsupported image type for `images[0]`. "
                             "Must be PIL.Image or NumPy array.")
        faces = self.app.get(experimental_img_bgr)

        # Ignore missing faces
        try:
            faceid_embed_exp = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        except IndexError:
            return 0.0

        return F.cosine_similarity(self.faceid_embed_gt, faceid_embed_exp).cpu().detach().item()
