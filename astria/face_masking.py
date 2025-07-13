import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize

import onnxruntime
import insightface
from insightface.app import FaceAnalysis

from facexlib.parsing import init_parsing_model

from astria_utils import CACHE_DIR

# Which labels from BiseNet are treated as background (we set them to 0)
# You can tune these if you only want certain parts of the face.
#  0: background
#  1: skin
#  2: left brow
#  3: right brow
#  4: left eye
#  5: right eye
#  6: nose
#  7: upper lip
#  8: inner mouth
#  9: lower lip
# 10: hair
# 11: left ear
# 12: right ear
# 13: earrings
# 14: neck
# 15: necklace
# 16: cloth
# 17: ??? (sometimes used for “hat,” or "hair," depending on model)
# 18: ??? (sometimes used for “mask” or unknown region)
BG_LABELS_WITH_NECK = [0, 7, 8, 9, 15, 16, 17, 18]
BG_LABELS_NO_NECK = [0, 7, 8, 9, 14, 15, 16, 17, 18]
BG_LABELS_WITH_NECK_AND_HAIR = [0, 7, 8, 15, 16, 18]

# Example color palette for 19 classes (0..18).
# You can pick any colors you like; these are just sample RGBs.
# Index i in this list means “label i” in the parsing map.
LABEL_COLORS = [
    (  0,   0,   0),  # 0: background
    (128,   0,   0),  # 1: skin
    (  0, 128,   0),  # 2: left brow
    (128, 128,   0),  # 3: right brow
    (  0,   0, 128),  # 4: left eye
    (128,   0, 128),  # 5: right eye
    (  0, 128, 128),  # 6: nose
    (128, 128, 128),  # 7: upper lip
    (192,   0,   0),  # 8: inner mouth
    (  0, 192,   0),  # 9: lower lip
    (192, 192,   0),  # 10: hair
    (  0,   0, 192),  # 11: left ear
    (192,   0, 192),  # 12: right ear
    (  0, 192, 192),  # 13: earring
    (192, 192, 192),  # 14: neck
    ( 64,   0,   0),  # 15: necklace
    (  0,  64,   0),  # 16: cloth
    ( 64,  64,   0),  # 17: hat? (depends on model)
    (  0,   0,   0),  # 18: ??? (some models differ)
]

def visualize_parsing(parsing_map: np.ndarray, pil_image: Image.Image, alpha=0.5) -> Image.Image:
    """
    Create a debug composite image:
      - Overlays each label in `parsing_map` with a distinct color.
      - Blends it with the original `pil_image`.
      - Draws a small legend showing label indices and their colors.
    """

    # Convert original image to NumPy (H,W,3) array
    orig_np = np.array(pil_image)
    h, w = orig_np.shape[:2]

    # Create a blank color overlay of the same size
    color_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # For each label in 0..max_label, assign a color to those pixels
    for label_id, color in enumerate(LABEL_COLORS):
        color_overlay[parsing_map == label_id] = color

    # Alpha blend overlay with original: result = alpha*orig + (1-alpha)*overlay
    blended = (alpha * orig_np + (1 - alpha) * color_overlay).astype(np.uint8)
    debug_pil = Image.fromarray(blended)

    # Draw legend
    draw = ImageDraw.Draw(debug_pil)
    # Try a default font or specify your own TTF
    # font = ImageFont.truetype("arial.ttf", 16)  # if you want a TTF

    # We’ll place the legend down the left side; shift as needed.
    y_offset = 10
    x_offset = 10
    box_size = 20

    for label_id in np.unique(parsing_map):
        ys, xs = np.where(parsing_map == label_id)
        if ys.size == 0:
            continue
        y, x = int(ys.mean()), int(xs.mean())
        # choose white or black text depending on underlying brightness
        pixel = orig_np[y, x].mean()
        text_color = (0,0,0) if pixel > 128 else (255,255,255)
        draw.text((x, y), str(label_id), fill=text_color)

    for label_id, color in enumerate(LABEL_COLORS):
        # (1) Draw a small color box
        box_coords = [
            (x_offset, y_offset),
            (x_offset + box_size, y_offset + box_size),
        ]
        draw.rectangle(box_coords, fill=color, outline=(255, 255, 255))
        
        # (2) Write text “Label X”
        text_pos = (x_offset + box_size + 5, y_offset)
        draw.text(text_pos, f"Label {label_id}", fill=(255, 255, 255)) # , font=font
        y_offset += box_size + 5  # move down for next entry

    return debug_pil

class FaceMaskGenerator:
    """
    A minimal class that:
      1) Uses InsightFace for face detection.
      2) Uses FaceXLib's BiseNet for face parsing to create a mask.
    """
    def __init__(
        self,
        device='cuda',
        providers=['CUDAExecutionProvider'],
        model_root='models'  # or wherever your FaceXLib parsing model is stored
    ):
        self.device = device

        # 1) Initialize face detection (InsightFace)
        # -----------------------------------------------------
        self.face_analysis = FaceAnalysis(providers=providers, root=CACHE_DIR)
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        # 2) Initialize face parsing (FaceXLib/BiseNet)
        # -----------------------------------------------------
        # This downloads or expects 'bisenet.onnx' in `model_root`
        self.parsing_model = init_parsing_model(
            model_name='bisenet', device=device, model_rootpath=model_root
        )

    @torch.no_grad()
    def get_face_mask(
        self,
        pil_image: Image.Image,
        expansion_factor=0.35,
        include_neck=True,
        include_hair=False,
    ) -> np.ndarray:
        """
        Detects the largest face in the image, parses it to get a binary mask
        (255 for face, 0 for background). Returns that mask at the original
        image resolution, as a numpy array of shape (H, W).
        """

        # Convert PIL (RGB) -> NumPy BGR for InsightFace
        orig_w, orig_h = pil_image.size
        img_bgr = np.array(pil_image)[:, :, ::-1].copy()

        # 1) Detect faces
        faces = self.face_analysis.get(img_bgr)
        if not faces:
            raise ValueError("No faces found in the image.")

        # Pick the largest face by bounding box area
        # faces = sorted(
        #     faces,
        #     key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
        #     reverse=True
        # )
        # x1, y1, x2, y2 = [int(v) for v in faces[0]['bbox']]
        # x1, y1 = max(0, x1), max(0, y1)
        # x2 = min(x2, img_bgr.shape[1])
        # y2 = min(y2, img_bgr.shape[0])
        faces = sorted(
            faces,
            key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
            reverse=True,
        )
        x1, y1, x2, y2 = [int(v) for v in faces[0]['bbox']]
        
        # ✅ Expand Bounding Box to Avoid Square Cropping
        face_width = x2 - x1
        face_height = y2 - y1
        expand_x = int(face_width * expansion_factor)
        expand_y = int(face_height * expansion_factor)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(orig_w, x2 + expand_x)
        y2 = min(orig_h, y2 + expand_y)

        # 2) Crop the largest face region
        face_crop = img_bgr[y1:y2, x1:x2]  # BGR
        face_orig_np = np.array(pil_image)[y1:y2, x1:x2]  # same shape as parsing_map
        face_orig_crop_image = Image.fromarray(face_orig_np)

        # 3) Run the face parsing model on the crop
        face_rgb = face_crop[:, :, ::-1].copy()
        face_tensor = to_tensor(face_rgb).unsqueeze(0).to(self.device)
        face_tensor = normalize(
            face_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # BiseNet returns shape (B, 19, H, W); we pick argmax
        parsing_out = self.parsing_model(face_tensor)[0]  # (1, 19, Hc, Wc)
        parsing_map = parsing_out.argmax(dim=1, keepdim=True)  # (1, 1, Hc, Wc)

        # 4) Upsample the parsing map to match the crop size
        parsing_map = F.interpolate(
            parsing_map.float(),
            size=(face_crop.shape[0], face_crop.shape[1]),
            mode='nearest'
        )[0, 0]  # shape (Hc, Wc), on CPU or GPU

        parsing_map = parsing_map.cpu().numpy().astype(np.uint8)

        # 5) Convert parsing_map into a binary mask:
        #    We'll set face pixels to 255, background to 0
        BG_LABELS = BG_LABELS_WITH_NECK
        if not include_neck:
            BG_LABELS = BG_LABELS_NO_NECK
        if include_hair:
            BG_LABELS = BG_LABELS_WITH_NECK_AND_HAIR

        mask_crop = np.full_like(parsing_map, 255, dtype=np.uint8)
        for bg_id in BG_LABELS:
            mask_crop[parsing_map == bg_id] = 0

        # 6) Place that mask back into the original coordinates
        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop

        return full_mask, parsing_map, face_orig_crop_image


if __name__ == '__main__':
    # 1) Initialize the generator
    face_masker = FaceMaskGenerator(device='cuda')  # or 'cpu'
    
    # 2) Load your input image
    input_path = '/ephemeral-data/models/2821958-faces/ngy4w7ol8xyuwkdcvn9q0b7smtsg-center-crop.png'
    pil_img = Image.open(input_path).convert("RGB")
    
    # 3) Run face detection + parsing
    full_mask, parsing_map, face_crop_img = face_masker.get_face_mask(
        pil_img,
        include_neck=True,
        include_hair=False,
    )
    
    # 4) Visualize the parsing on the cropped face region
    debug_vis = visualize_parsing(
        parsing_map,
        face_crop_img,
        alpha=0.6
    )
    debug_vis.save("parsing_debug.png")
    debug_vis.show()
    print("Saved visualization to parsing_debug.png")
    
    # 5) Now produce versions with different BG_LABELS masked out
    bg_variants = {
        "with_neck": BG_LABELS_WITH_NECK,
        "no_neck": BG_LABELS_NO_NECK,
        "with_neck_and_hair": BG_LABELS_WITH_NECK_AND_HAIR,
    }
    
    # Convert the PIL crop to a NumPy array once
    crop_np = np.array(face_crop_img)
    Hc, Wc = parsing_map.shape
    
    for variant, bg_labels in bg_variants.items():
        # build a boolean mask: True where label is background
        bg_mask = np.isin(parsing_map, bg_labels)
        
        # apply: zero out background pixels
        out_np = crop_np.copy()
        out_np[bg_mask] = 0
        
        # save
        out_pil = Image.fromarray(out_np)
        fname = f"mask_{variant}.png"
        out_pil.save(fname)
        print(f"Saved masked face ({variant}) to {fname}")