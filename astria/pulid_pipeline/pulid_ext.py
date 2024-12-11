import sys
sys.path.append("astria/pulid_pipeline")
from typing import Final
from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from huggingface_hub import hf_hub_download, snapshot_download
import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import numpy as np
from PIL import Image
from .pulid.encoders_transformer import IDFormer, PerceiverAttentionCA
import os
import glob
from safetensors.torch import load_file
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
import cv2
from pathlib import Path
import sys
import onnxruntime
from astria_utils import CUDA_VISIBLE_DEVICES

###
# Modified by huggingface/twodgirl.
# License: apache-2.0
# License and config from huggingface/guozinan.
# Any copy of this file must retain the license docstring.

# ======================
# Final Constants
# ======================
PULID_VERSION: Final[str] = 'v0.9.1'
PULID_MODEL_FILENAME_PATTERN: Final[str] = 'pulid_flux_{version}.safetensors'
PULID_REPO_ID: Final[str] = 'guozinan/PuLID'
PULID_LOCAL_DIR: Final[str] = 'models'

EVA_CLIP_MODEL_NAME: Final[str] = 'EVA02-CLIP-L-14-336'
EVA_CLIP_LIBRARY_NAME: Final[str] = 'eva_clip'

ANTEL_REPO_ID: Final[str] = 'DIAMONIK7777/antelopev2'
ANTEL_MODEL_FILE: Final[str] = 'glintr100.onnx'

FACE_DET_MODEL: Final[str] = 'retinaface_resnet50'
FACE_SAVE_EXT: Final[str] = 'png'
FACE_PARSING_MODEL: Final[str] = 'bisenet'

BG_LABELS: Final[list[int]] = [0, 16, 18, 7, 8, 9, 14, 15]

ID_IMAGE_MAX_EDGE: Final[int] = 1024
CLIP_IMAGE_SIZE: Final[int] = 336  # Based on EVA02-CLIP-L-14-336 model
TRUE_CFG_TOL: Final[float] = 1e-2

# Seems like when env variable CUDA_VISIBLE_DEVICES is defined, the devices onnx sees is already filtered
# PROVIDERS = [('CUDAExecutionProvider', {'device_id': CUDA_VISIBLE_DEVICES})]
PROVIDERS = ['CUDAExecutionProvider']


def resize_numpy_image_long(image, resize_long_edge=ID_IMAGE_MAX_EDGE):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


class PatchedFaceAnalysis(FaceAnalysis):
    '''
    Class to deal with FaceAnalysis doing things with paths they should not.
    '''
    def __init__(self, model_dir, name='default', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        # Instead of calling ensure_available, we directly set model_dir:
        self.model_dir = model_dir
        onnx_files = glob.glob(os.path.join(self.model_dir, '**', '*.onnx'), recursive=True)
        onnx_files = sorted(onnx_files)
        from insightface.model_zoo import model_zoo
        for onnx_file in onnx_files:
            print('loading model:', onnx_file, kwargs)
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


class PuLModel(nn.Module):
    double_interval = 2
    single_interval = 4

    def __init__(self, device, dtype):
        super().__init__()
        self.pulid_encoder = IDFormer().to(device, dtype)
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(device, dtype) for _ in range(num_ca)
        ])


class PuLID:
    def __init__(
        self,
        device='cuda',
        dtype=torch.bfloat16,
        version=PULID_VERSION,
        local_dir=PULID_LOCAL_DIR,
    ):
        ANTEL_LOCAL_DIR: Final[str] = f'{local_dir}/antelopev2'
        self.antel_local_dir = ANTEL_LOCAL_DIR
        encoder_path = PULID_MODEL_FILENAME_PATTERN.format(version=version)
        if not os.path.exists(encoder_path):
            hf_hub_download(PULID_REPO_ID, encoder_path, local_dir=local_dir)
        self.model = PuLModel(device, dtype)
        self.model.load_state_dict(load_file(os.path.join(local_dir, encoder_path)))
        self.face_helper = PuLID.init_face_helper(device)
        (
            self.clip_vision_model,
            self.eva_transform_mean,
            self.eva_transform_std,
        ) = PuLID.init_clip(device, dtype)
        if not os.path.exists(ANTEL_LOCAL_DIR):
            snapshot_download(ANTEL_REPO_ID, local_dir=ANTEL_LOCAL_DIR)
        self.app, self.handler_ante = self.init_insightface()

    @staticmethod
    def init_clip(device, dtype):
        model, _, _ = create_model_and_transforms(EVA_CLIP_MODEL_NAME, EVA_CLIP_LIBRARY_NAME, force_custom_clip=True)
        model = model.visual
        clip_vision_model = model.to(device, dtype=dtype)
        eva_transform_mean = getattr(clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        eva_transform_mean = eva_transform_mean
        eva_transform_std = eva_transform_std

        return clip_vision_model, eva_transform_mean, eva_transform_std

    @staticmethod
    def init_face_helper(device):
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=FACE_DET_MODEL,
            save_ext=FACE_SAVE_EXT,
            device=device,
        )
        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name=FACE_PARSING_MODEL, device=device)

        return face_helper

    def init_insightface(self):
        app = PatchedFaceAnalysis(self.antel_local_dir, name='antelopev2', root=self.antel_local_dir, providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=(640, 640))

        model_loc = f'{self.antel_local_dir}/{ANTEL_MODEL_FILE}'
        # next(list(filter()))
        if not Path(model_loc).exists():
            print('Could not find insightface model in ', model_loc)
            sys.exit(1)

        # Ensure the 'models' directory exists inside self.model_dir
        models_dir = os.path.join(self.antel_local_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Recursively find all .onnx files under self.model_dir
        onnx_files = glob.glob(os.path.join(self.antel_local_dir, '**', '*.onnx'), recursive=True)

        # Create symbolic links in self.model_dir/models for each .onnx file found
        antel_model_file_loc = None
        for onnx_file in onnx_files:
            link_path = os.path.join(models_dir, os.path.basename(onnx_file))
            if ANTEL_MODEL_FILE in onnx_file:
                antel_model_file_loc = link_path
            # Check if symlink already exists, if not, create it
            if not os.path.exists(link_path):
                os.symlink(onnx_file, link_path)

        handler = insightface.model_zoo.get_model(antel_model_file_loc, root=self.antel_local_dir, providers=PROVIDERS)
        handler.prepare(ctx_id=0)

        return app, handler

    def setup(self, transformer):
        transformer.pulid_ca = self.model.pulid_ca
        transformer.pulid_double_interval = self.model.double_interval
        transformer.pulid_single_interval = self.model.single_interval

    @staticmethod
    def resize_image(image: Image.Image, long_edge=768):
        w, h = image.size
        if max(w, h) <= long_edge:
            return np.array(image)
        k = long_edge / max(h, w)
        w = int(w * k)
        h = int(h * k)

        return np.array(image.resize((w, h)))

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding_by_pil(self, image: Image.Image, true_cfg=1.0):
        use_true_cfg = abs(true_cfg - 1.0) > TRUE_CFG_TOL
        id_image = self.resize_image(image, ID_IMAGE_MAX_EDGE)
        id_embeddings, uncond_id_embeddings = self.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        return id_embeddings, uncond_id_embeddings

    @torch.no_grad()
    def get_id_embedding(
        self,
        image,
        cal_uncond=False,
        device='cuda',
        dtype=torch.bfloat16,
    ):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()

        if isinstance(image, Image.Image):
            image = np.array(image)  # shape: (H, W, 3), dtype: uint8, range [0,255]

        image = resize_numpy_image_long(image)

        # RGB -> BGR
        image_bgr = image[:, :, ::-1].copy()

        # Get antelopev2 embedding.
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info,
                               key=lambda x: (x['bbox'][2] - x['bbox'][0]) *
                                             (x['bbox'][3] - x['bbox'][1]))[-1]
            id_ante_embedding = face_info['embedding']
        else:
            id_ante_embedding = None

        # Using facexlib to detect and align face.
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]

        # In case insightface didn't detect face.
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)
        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # Convert BGR to RGB.
        input_tensor = torch.from_numpy(align_face[:, :, ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)
        normalized = normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        parsing_out = self.face_helper.face_parse(normalized)[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg = sum(parsing_out == i for i in BG_LABELS).bool()
        white_image = torch.ones_like(input_tensor)
        # Only keep the face features.
        face_features_image = torch.where(bg, white_image, self.to_gray(input_tensor))

        # Transform img before sending to eva-clip-vit.
        face_features_image = resize(face_features_image, CLIP_IMAGE_SIZE, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        (id_cond_vit,
         id_vit_hidden) = self.clip_vision_model(face_features_image.to(dtype),
                                                 return_all_features=False,
                                                 return_hidden=True,
                                                 shuffle=False)
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
        id_embedding = self.model.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.model.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding

if __name__ == '__main__':
    pulid_model = PuLID(local_dir='/home/user/storage/hf_cache/pulid')
    img = Image.open('test/margarethamilton.jpg')
    embed, _ = pulid_model.get_id_embedding(img)
    print(embed.shape)
