from functools import partial
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms
from insightface.app import FaceAnalysis
import torch, logging, os, cv2
import numpy as np
from PIL import Image
from helpers.training.state_tracker import StateTracker
import torch.nn.functional as F
import tqdm

logger = logging.getLogger("ModelEvaluator")
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

model_evaluator_map = {
    "clip": "CLIPModelEvaluator",
    "face": "FaceModelEvaluator",
}


class ModelEvaluator:
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        raise NotImplementedError(
            "Subclasses is incomplete, no __init__ method was found."
        )

    def evaluate(self, images, prompts, **kwargs):
        raise NotImplementedError("Subclasses should implement the evaluate() method.")

    @staticmethod
    def from_config(args, **kwargs):
        """Instantiate a ModelEvaluator from the training config, if set to do so."""
        if not StateTracker.get_accelerator().is_main_process:
            return None
        if (
            args.evaluation_type is not None
            and args.evaluation_type.lower() != ""
            and args.evaluation_type.lower() != "none"
        ):
            model_evaluator = model_evaluator_map[args.evaluation_type]
            return globals()[model_evaluator](
                args.pretrained_evaluation_model_name_or_path,
                **kwargs
            )

        return None


class CLIPModelEvaluator(ModelEvaluator):
    def __init__(
        self, pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    ):
        self.clip_score_fn = partial(
            clip_score, model_name_or_path=pretrained_model_name_or_path
        )
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def evaluate(self, images, prompts):
        # Preprocess images
        images_tensor = torch.stack([self.preprocess(img) * 255 for img in images])
        # Compute CLIP scores
        result = self.clip_score_fn(images_tensor, prompts).detach().cpu()

        return result


class FaceModelEvaluator(ModelEvaluator):
    def __init__(self, pretrained_model_name_or_path='buffalo_l', **kwargs):
        # ── 1) Initialize InsightFace ───────────────────────────────
        self.app = FaceAnalysis(
            name=pretrained_model_name_or_path or 'buffalo_l',
            root='./faceanalysis',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # ── 2) Build baseline embedding with retry logic ────────────
        baseline_images = kwargs["baseline_images"]
        embeds = []
        for img_path in tqdm.tqdm(baseline_images, desc="Generating baseline face embeds..."):
            img_bgr = cv2.imread(img_path)
            faces = self._detect_with_retry(img_bgr)
            if not faces:
                continue
            embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            embeds.append(embed)

        if not embeds:
            raise RuntimeError("No faces detected in any baseline image!")

        # average baseline
        self.faceid_embed_gt = torch.stack(embeds, dim=0).mean(0)

    def _detect_with_retry(self, img_bgr: np.ndarray) -> list:
        """
        Try face detection on img_bgr; if no faces found, 'zoom out'
        by 16px increments (shrink + pad) up to 5 times.
        """
        # 1) First try at original resolution
        faces = self.app.get(img_bgr)
        if faces:
            return faces

        h0, w0 = img_bgr.shape[:2]
        for i in range(0, 5):
            shrink = 16 * i
            new_h = max(1, h0 - shrink)
            new_w = max(1, w0 - shrink)
            # shrink the image
            resized = cv2.resize(img_bgr, (new_w, new_h))
            # compute padding to restore original size
            delta_h = h0 - new_h
            delta_w = w0 - new_w
            top    = delta_h // 2
            bottom = delta_h - top
            left   = delta_w // 2
            right  = delta_w - left
            padded = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            faces = self.app.get(padded)
            if faces:
                return faces

        return []

    def evaluate(self, images, prompts, **kwargs):
        assert len(images) == 1, 'too many images passed to FaceModelEvaluator'

        # ── 1) Prepare a BGR numpy image ─────────────────────────────
        img0 = images[0]
        if isinstance(img0, Image.Image):
            arr = np.array(img0.convert("RGB"))
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif isinstance(img0, np.ndarray):
            img_bgr = img0
        else:
            raise ValueError("Unsupported image type; must be PIL.Image or NumPy array")

        # ── 2) Detect (with zoom‐out retries) ────────────────────────
        faces = self._detect_with_retry(img_bgr)
        if not faces:
            return 0.0

        # ── 3) Embed & compare ───────────────────────────────────────
        embed_exp = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        score = F.cosine_similarity(self.faceid_embed_gt, embed_exp, dim=1)
        return score.cpu().item()