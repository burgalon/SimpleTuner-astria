# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random

import torch
import torchvision.transforms as T
import numpy as np
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from PIL import Image


from PIL import Image
from collections import Counter


def get_most_common_color(image):
    """
    Returns the most common color in the given image.
    If the image has many colors, it will downsample to speed up the process.
    """
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    
    # You can try to get all colors; if the image is very large,
    # you might want to resize it for performance.
    # Here, we try the full image:
    max_colors = image.width * image.height
    colors = image.getcolors(max_colors)
    
    # If too many colors, downsample
    if colors is None:
        small_image = image.resize((image.width // 10, image.height // 10))
        colors = small_image.getcolors(small_image.width * small_image.height)
    
    # Get the most common color (the one with the highest count)
    most_common_color = max(colors, key=lambda x: x[0])[1]
    return most_common_color


def scale_long_edge_and_pad(img: Image.Image, sz=1024) -> Image.Image:
    # Open the image
    original_width, original_height = img.size

    # Compute scale factor: scale long edge to sz pixels.
    scale_factor = sz / max(original_width, original_height)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image using a high-quality filter.
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Get the most common color from the original image.
    common_color = get_most_common_color(img)

    # Create a new 1024x1024 image filled with the common color.
    new_img = Image.new("RGB", (sz, sz), common_color)

    # Calculate the top-left coordinates to center the resized image.
    left = (sz - new_width) // 2
    top = (sz - new_height) // 2

    # Paste the resized image onto the new background.
    new_img.paste(resized_img, (left, top))

    # Save the final image.
    return new_img


def random_crop_pil(image, crop_size):
    # image: a PIL.Image instance
    # crop_size: tuple (crop_width, crop_height)
    image_width, image_height = image.size
    crop_width, crop_height = crop_size

    if image_width < crop_width or image_height < crop_height:
        raise ValueError("The image is smaller than the crop size.")

    # Determine the maximum x and y coordinates for the top-left corner
    max_x = image_width - crop_width
    max_y = image_height - crop_height

    # Randomly choose the top-left coordinates for the crop
    x0 = random.randint(0, max_x)
    y0 = random.randint(0, max_y)

    # Define the box to crop: (left, upper, right, lower)
    crop_box = (x0, y0, x0 + crop_width, y0 + crop_height)
    cropped_image = image.crop(crop_box)

    return cropped_image


def edit_preprocess(processor, device, edit_image, edit_mask):
    if edit_image is None or processor is None:
        return edit_image
    processor = Config(cfg_dict=processor, load=False)
    processor = ANNOTATORS.build(processor).to(device)
    new_edit_image = processor(np.asarray(edit_image))
    processor = processor.to("cpu")
    del processor
    new_edit_image = Image.fromarray(new_edit_image)
    return Image.composite(new_edit_image, edit_image, edit_mask)

class ACEPlusImageProcessor():
    def __init__(self, max_aspect_ratio=4, d=16, max_seq_len=1024):
        self.max_aspect_ratio = max_aspect_ratio
        self.d = d
        self.max_seq_len = max_seq_len
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def image_check(self, image):
        if image is None:
            return image
        # preprocess
        W, H = image.size
        if H / W > self.max_aspect_ratio:
            image = T.CenterCrop([int(self.max_aspect_ratio * W), W])(image)
        elif W / H > self.max_aspect_ratio:
            image = T.CenterCrop([H, int(self.max_aspect_ratio * H)])(image)
        return self.transforms(image)


    def preprocess(self,
                   reference_image=None,
                   edit_image=None,
                   edit_mask=None,
                   height=1024,
                   width=1024,
                   repainting_scale = 1.0):
        reference_image = self.image_check(reference_image)
        edit_image = self.image_check(edit_image)
        # for reference generation
        if edit_image is None:
            edit_image = torch.zeros([3, height, width])
            edit_mask = torch.ones([1, height, width])
        else:
            edit_mask = np.asarray(edit_mask)
            edit_mask = np.where(edit_mask > 128, 1, 0)
            edit_mask = edit_mask.astype(
                np.float32) if np.any(edit_mask) else np.ones_like(edit_mask).astype(
                np.float32)
            edit_mask = torch.tensor(edit_mask).unsqueeze(0)

        edit_image = edit_image * (1 - edit_mask * repainting_scale)


        out_h, out_w = edit_image.shape[-2:]

        assert edit_mask is not None
        if reference_image is not None:
        # align height with edit_image
            _, H, W = reference_image.shape
            _, eH, eW = edit_image.shape
            scale = eH / H
            tH, tW = eH, int(W * scale)
            reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(reference_image)
            edit_image = torch.cat([reference_image, edit_image], dim=-1)
            edit_mask = torch.cat([torch.zeros([1, reference_image.shape[1], reference_image.shape[2]]), edit_mask], dim=-1)
            slice_w = reference_image.shape[-1]
        else:
            slice_w = 0

        H, W = edit_image.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / self.d) * (W / self.d))))
        rH = int(H * scale) // self.d * self.d  # ensure divisible by self.d
        rW = int(W * scale) // self.d * self.d
        slice_w = int(slice_w * scale) // self.d * self.d

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)

        return edit_image, edit_mask, out_h, out_w, slice_w


    def postprocess(self, image, slice_w, out_w, out_h):
        w, h = image.size
        if slice_w > 0:
            output_image = image.crop((slice_w + 20, 0, w, h))
            output_image = output_image.resize((out_w, out_h))
        else:
            output_image = image
        return output_image