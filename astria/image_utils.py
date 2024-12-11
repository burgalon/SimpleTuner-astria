import base64
from io import BytesIO

# do not remove - this import registers AV1 handler
import pillow_avif
from pillow_heif import register_heif_opener
register_heif_opener()

import os
import cv2
import numpy as np
import requests
from PIL import Image, ImageFile
import PIL.Image
from typing import Union
import os
s3_session = requests.Session()
retries = requests.packages.urllib3.util.retry.Retry(total=20, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 403, 404, 401], raise_on_status=True)
s3_session.mount('', requests.adapters.HTTPAdapter(max_retries=retries))

ImageFile.LOAD_TRUNCATED_IMAGES = True

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def io2img(url, convert = 'RGB') -> Image.Image:
    try:
        image = Image.open(url)
        if convert:
            if convert == 'L' and image.mode == 'RGBA':
                # convert transparent to black
                print("Converting RGBA to L")
                image = Image.composite(Image.new('RGB', image.size, (255, 255, 255)), Image.new('RGB', image.size, (0, 0, 0)), image)
        image = image.convert(convert)
        # clear ICCProfile to avoid PIL warnings
        image.info.pop('icc_profile', None)
    except Exception as e:
        print(f"Failed to open image {url}")
        raise e
    image = _apply_exif_orientation(image)
    return image

def url2img(url, convert = 'RGB') -> Image.Image:
    if url.startswith("http"):
        print(f"Downloading {url}")
        for i in range(10):
            try:
                response = s3_session.get(url, stream=True)
                response.raise_for_status()
                return io2img(response.raw, convert)
            except Exception as e:
                print(f"Failed to download {url} on attempt {i+1}: {e}")
        raise Exception(f"Failed to download {url}")
    else:
        image_data = url
        if os.path.isfile(url):
            with open(url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read())
        image_data = io2img(BytesIO(base64.b64decode(image_data)), convert)
        return image_data

def file2base64(filename) -> str:
    """ For testing """
    with open(filename, "rb") as file:
        data = file.read()
    return base64.b64encode(data).decode('utf-8')

def ndarray2base64(nd_image) -> str:
    """ For testing """
    img = Image.fromarray(cv2.cvtColor(nd_image, cv2.COLOR_BGR2RGB))
    return pil2base64(img)

def b64_to_single_img(image: str) -> "np.ndarray":
    return cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), -1)

def b64_to_single_pil(image: str) -> Image:
    return Image.open(BytesIO(base64.b64decode(image)))

def b64_to_img(base64_images: list) -> list["np.ndarray"]:
    return [b64_to_single_img(image) for image in base64_images]

def pil2ndarray(image: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# PNG is used for multiperson to avoid deteriorating the image quality over passes and also for mask
def pil2base64(image: Image, format="PNG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format, optimize=False, compression=0)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_image(image: Union[str, PIL.Image.Image], convert = 'RGB') -> PIL.Image.Image:
    """
    Args:
    Loads `image` to a PIL Image.
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        image = url2img(image, convert)
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    return image

def save_img(image: Image, fn: str, format="PNG") -> str:
    return image.save(fn, format=format, optimize=False, compression=0)
