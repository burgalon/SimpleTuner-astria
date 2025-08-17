import math

import requests
import requests
from PIL import Image
import io
import os
import time
import uuid
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
import rollbar
from PIL import Image
from astria_utils import JsonObj, MODELS_DIR

if os.environ.get('MOCK_SERVER') or os.environ.get('DEBUG') == 'test':
    from astria_mock_server import FASHN_API_KEY, BAKE_API_KEY
else:
    from astria_server import FASHN_API_KEY, BAKE_API_KEY
from image_utils import pil2base64, load_image
from request_session import session

LOWER_BODY_CATEGORIES = ['pants', 'shorts', 'skirt', 'jeans', 'trousers']
UPPER_BODY_CATEGORIES = ['shirt', 't-shirt', 'jacket', 'sweater', 'hoodie', 'coat', 'parka']
OTHER_CLOTHING_CATEGORIES = [
    'clothing', 'swimming suit', 'bathing suit', 'dress', 'blouse', 'vest', 'sweatshirt',
    'sweatpants', 'suit', 'uniform', 'costume', 'robe', 'kimono', 'tunic', 'gown', 'overalls',
    'coveralls', 'jumpsuit', 'romper', 'onesie', 'pajamas', 'nightgown', 'nightshirt',
    'nightwear', 'nightie'
]
VTON_CATEGORIES = LOWER_BODY_CATEGORIES + UPPER_BODY_CATEGORIES + OTHER_CLOTHING_CATEGORIES

HEADERS = {"Authorization": f"Key {FASHN_API_KEY}", "Content-Type": "application/json"}
HEADERS_BAKE = {"Authorization": f"Key {BAKE_API_KEY}", "Content-Type": "application/json"}
FASHN_BASE_URL_V1 = 'https://api.fashn.ai/v1'
FASHN_BASE_URL_NIGHTLY = 'https://api.fashn.ai/nightly'

s3_boto = None
if not os.environ.get('MOCK_SERVER', False):
    s3_boto = boto3.client('s3',
        endpoint_url=os.environ.get('R2_ENDPOINT_URL_S3'),
        aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
    )
S3_BUCKET = 'sdbooth2-production'
PUBLIC_BUCKET_URL = 'https://mp.astria.ai/'

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB
MAX_DIMENSION = 6000

def crop_to_aspect_ratio(result_image: Image.Image, image: Image.Image) -> Image.Image:
    # if orig `image` is narrower
    print(f"Cropping result image {result_image.width}x{result_image.height} to match original aspect ratio {image.width}x{image.height}")
    if result_image.width / result_image.height > image.width / image.height:
        # crop the sides to match the original aspect ratio
        target_width = result_image.height * image.width / image.height
        # crop extra 1 pixels from each side which seems to be padded over
        margin = math.floor(result_image.width - target_width) // 2 + 1
        print(f"Cropping sides to new {target_width=} margin={margin}")
        result_image = result_image.crop((margin, 0, result_image.width - margin, result_image.height))
    return result_image

class VtonMixin:
    def __init__(self):
        pass

    def vton(self, images: List[Image.Image], prompt: JsonObj):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.vton_image if os.environ.get('USE_FASHN') or prompt.vton_model == 'fashn' else self.vton_image_bake, image, prompt)
                for image in images
            ]

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # if os.environ.get('DEBUG'):
        #     for i, image in enumerate(results):
        #         image.save(MODELS_DIR + f"/{prompt.id}-{i}-after-vton.jpg")

        return results

    def _prepare_and_upload_image(self,
            image_source: Image.Image | str,
            image_type: str,
            prompt_id: str,
            s3_client,
            bucket: str,
            public_url_prefix: str
    ) -> tuple[str | None, str | None]:
        """
        Prepares an image for the VTON API by downloading, resizing, and uploading it.

        This function will:
        1. Download the image if a URL is provided.
        2. Resize the image if it exceeds 2MB or 6000x6000 pixels.
        3. Upload the processed image to a temporary S3 location.
        4. Verify the uploaded image is publicly accessible.

        Args:
            image_source: A PIL Image object or a URL string.
            image_type: A string descriptor (e.g., 'human', 'garment').
            prompt_id: The ID of the current prompt for unique naming.
            s3_client: The Boto3 S3 client.
            bucket: The S3 bucket name.
            public_url_prefix: The public base URL for the S3 bucket.

        Returns:
            A tuple containing the public URL and the S3 key of the uploaded image,
            or (None, None) if an error occurs.
        """
        try:
            # Load image from URL or use the provided PIL Image object
            if isinstance(image_source, str):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            else:
                img = image_source
        except requests.RequestException as e:
            print(f"P={prompt_id} Failed to download {image_type} image: {e}")
            return None, None

        # Check if resizing is needed
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        needs_resize = (img_byte_arr.tell() > MAX_FILE_SIZE_BYTES or
                        img.width > MAX_DIMENSION or img.height > MAX_DIMENSION)

        if needs_resize:
            print(f"Resizing {image_type} image for prompt {prompt_id}. "
                  f"Original size: {img_byte_arr.tell() / 1e6:.2f}MB, "
                  f"Dimensions: {img.width}x{img.height}")

            # Resize based on dimensions while preserving aspect ratio
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

            # Save to PNG and re-check size
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True)

            # If still too large, convert to JPEG for better compression
            if img_byte_arr.tell() > MAX_FILE_SIZE_BYTES:
                print("Image still too large after PNG optimization, converting to JPEG.")
                img_byte_arr = io.BytesIO()
                # JPEG doesn't support alpha, so convert to RGB
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(img_byte_arr, format='JPEG', quality=90) # High quality JPEG

        img_byte_arr.seek(0)

        # Upload the final image data to S3
        random_suffix = uuid.uuid4().hex[:8]
        key = f"tmp/vton-{prompt_id}-{image_type}-{random_suffix}.png"
        print(f"Uploading {image_type} image to S3 with key: {key}")
        s3_client.upload_fileobj(img_byte_arr, bucket, key)
        url = public_url_prefix + key

        # Verify that the image is accessible via the public URL
        for i in range(10):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Successfully verified {image_type} image at {url}")
                    return url, key
            except requests.RequestException as e:
                print(f"Failed to fetch {image_type} image {url} (attempt {i+1}/10): {e}")
                time.sleep(0.5)

        print(f"P={prompt_id} Could not verify {image_type} image upload at {url}")
        s3_client.delete_object(Bucket=bucket, Key=key) # Cleanup failed upload
        return None, None

    def vton_image_bake(self, image: Image.Image, prompt: JsonObj):
        for tune in prompt.tunes:
            if tune.name not in VTON_CATEGORIES or not tune.face_swap_images:
                continue
            prompt.hires_denoising_strength = 0.1

            category = (
                'top' if tune.name in UPPER_BODY_CATEGORIES
                else 'bottom' if tune.name in LOWER_BODY_CATEGORIES
                else 'full'
            )
            cfg_scale = prompt.vton_cfg_scale or 0.5

            garment_photo_type = 'model' if 'model' in tune.title else 'flat-lay' if 'flat-lay' in tune.title else 'auto'
            print(f"Running vton for {tune.name=} {category=} {cfg_scale=} {garment_photo_type=}")

            # Prepare human and garment images by uploading them to temporary S3 locations
            human_url, human_key = self._prepare_and_upload_image(
                image_source=image,
                image_type='human',
                prompt_id=prompt.id,
                s3_client=s3_boto,
                bucket=S3_BUCKET,
                public_url_prefix=PUBLIC_BUCKET_URL
            )

            garment_url, garment_key = self._prepare_and_upload_image(
                image_source=tune.face_swap_images[0],
                image_type='garment',
                prompt_id=prompt.id,
                s3_client=s3_boto,
                bucket=S3_BUCKET,
                public_url_prefix=PUBLIC_BUCKET_URL
            )

            # Step 1: Run the model
            response = requests.post('https://api.alphabake.io/api/v2/tryon/', json={
                'human_url': human_url,
                'garment_url': garment_url,
                'garment_type': category,
                'mode': prompt.vton_quality or 'fast', #'fast(5s, 1 credits, 768)' or 'quality(10s, 2 credits, 1024)'
                'garment_guidance': cfg_scale, #0.0 to 1.0 (optional, default of 0.5 is good for most cases, reduce if needed)
                'process_asset': 'tryon', #'tryon' or 'garment' or 'human', credits will be deducted for tryon
                'human_zoom_in': 'false' #'true' or 'false', default is true
            }, headers=HEADERS_BAKE)

            print(f"VTON response: {response.status_code} {response.text}")
            # delete the temporary image from S3
            if s3_boto is not None:
                s3_boto.delete_object(Bucket=S3_BUCKET, Key=human_key)
                s3_boto.delete_object(Bucket=S3_BUCKET, Key=garment_key)


            response_data = response.json()
            if 'tryon_id' not in response_data:
                print(f"P={prompt.id} Failed to start VTON process: {response_data}")
                return image

            # Step 2: Poll for the status
            tryon_id = response_data['tryon_id']
            print(f"Polling {tryon_id=} for VTON status...")
            # for quick testing
            # status_url = "https://queue.fal.run/fashn/tryon/requests/8510818b-d43e-4c4a-a6ab-38d5bbc80052/status"
            start_time = time.time()
            for _ in range(240):
                status_response = session.post('https://api.alphabake.io/api/v2/tryon_status/', json={
                    'tryon_id': tryon_id,
                }, headers=HEADERS_BAKE)
                status_data = status_response.json()

                # https://docs.fal.ai/model-endpoints/queue/
                if 'status' not in status_data:
                    print(f"VTON status response missing status: {status_data}")
                    continue
                if status_data['status'] in ['done']:
                    break
                time.sleep(0.5)

            if status_data['status'] != 'done':
                rollbar_uuid=rollbar.report_message(f"P={prompt.id} Failed to get response from VTON: {status_data}", "error")
                print(f"P={prompt.id} Failed to get response from VTON: {status_data} {rollbar_uuid=}")
                return image


            # Step 3: Fetch the output
            image_url = status_data['s3_url']
            print(f"Successfully completed VTON. Fetching image from {image_url}. Time={time.time() - start_time:.2f}s")
            result_image = load_image(image_url)
            result_image = crop_to_aspect_ratio(result_image, image)
            image = result_image

            # API returns always centered 768x1024 padded by white background -
        return image

    def vton_image(self, image: Image.Image, prompt: JsonObj):
        for tune in prompt.tunes:
            if tune.name not in VTON_CATEGORIES or not tune.face_swap_images:
                continue
            prompt.hires_denoising_strength = 0.1

            category = (
                'tops' if tune.name in UPPER_BODY_CATEGORIES
                else 'bottoms' if tune.name in LOWER_BODY_CATEGORIES
                else 'one-pieces'
            )
            cfg_scale = prompt.vton_cfg_scale or 2

            garment_photo_type = 'model' if 'model' in tune.title else 'flat-lay' if 'flat-lay' in tune.title else 'auto'
            print(f"Running vton for {tune.name=} {category=} {cfg_scale=} {garment_photo_type=}")

            # Step 1: Run the model
            base_url = 'https://queue.fal.run/fal-ai/fashn/tryon/v1.6'
            response = session.post(base_url, json={
                'model_image': "data:image/png;base64, " + pil2base64(image),
                'garment_image': tune.face_swap_images[0],
                'category': category,
                'guidance_scale': cfg_scale,
                'garment_photo_type': garment_photo_type,
                'nsfw_filter': False,
                'restore_clothes': category != 'one-pieces',
                'segmentation_free': False,
            }, headers=HEADERS)

            response_data = response.json()
            if 'status' not in response_data:
                print(f"P={prompt.id} Failed to start VTON process: {response_data.get('error', 'Unknown error')}")
                return image

            # Step 2: Poll for the status
            status_url = response_data['status_url']
            # for quick testing
            # status_url = "https://queue.fal.run/fashn/tryon/requests/8510818b-d43e-4c4a-a6ab-38d5bbc80052/status"
            start_time = time.time()
            for _ in range(120):
                status_response = session.get(status_url, headers=HEADERS)
                status_data = status_response.json()

                # https://docs.fal.ai/model-endpoints/queue/
                if 'status' not in status_data:
                    print(f"VTON status response missing status: {status_data}")
                    continue
                if status_data['status'] in ['COMPLETED']:
                    break
                time.sleep(0.5)

            if status_data['status'] != 'COMPLETED':
                uuid=rollbar.report_message(f"P={prompt.id} Failed to get response from VTON: {status_data.get('DETAIL', 'Unknown error')}", "error")
                print(f"P={prompt.id} Failed to get response from VTON: {status_data.get('error', 'Unknown error')} {uuid=}")
                return image

            result_data = requests.get(status_data['response_url'], headers=HEADERS).json()
            if 'images' not in result_data:
                print(f"P={prompt.id} VTON response missing images: {result_data}")
                if result_data['detail'] and isinstance(result_data['detail'], str):
                    result_data = result_data['detail']
                elif result_data['detail'] and isinstance(result_data['detail'], list):
                    result_data = ", ".join([detail['msg'] for detail in result_data['detail']])
                elif result_data['detail'] and result_data['detail']['message']:
                    result_data = result_data['detail']['message']
                uuid=rollbar.report_message(f"P={prompt.id} Failed to get response from VTON: {result_data}", "error")
                print(f"P={prompt.id} Failed to get images from VTON: {result_data} {uuid=}")
                return image

            # Step 3: Fetch the output
            image_url = result_data['images'][0]['url']
            print(f"Successfully completed VTON. Fetching image from {image_url}. Time={time.time() - start_time:.2f}s")
            image = load_image(image_url)

        return image

if __name__ == "__main__":
    result_image = crop_to_aspect_ratio(
        load_image('https://alpha-bake-loras.s3.amazonaws.com/Scale/human_processed/e7c518e3-4502-4a3d-9807-4c57de4f3dc2_final-asset.jpg?AWSAccessKeyId=AKIA3CTOLBCIXYCF4B2Q&Signature=NCiE58Eg25il%2F52lIXMCX5vE700%3D&Expires=1785502246'),
        load_image('https://mp.astria.ai/z7fvyofa77w4w0b5f9hbu4n43azd'),
    )
    result_image.save('/data/models/result_image.jpg')
