import os
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from astria_utils import JsonObj, MODELS_DIR
if os.environ.get('MOCK_SERVER'):
    from astria_mock_server import FASHN_API_KEY
else:
    from astria_server import FASHN_API_KEY
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

HEADERS = {"Authorization": f"Bearer {FASHN_API_KEY}"}
FASHN_BASE_URL_V1 = 'https://api.fashn.ai/v1'
FASHN_BASE_URL_NIGHTLY = 'https://api.fashn.ai/nightly'


class VtonMixin:
    def __init__(self):
        pass

    def vton(self, images: List[Image.Image], prompt: JsonObj):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.vton_image, image, prompt)
                for image in images
            ]

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        if os.environ.get('DEBUG'):
            for i, image in enumerate(results):
                image.save(MODELS_DIR + f"/{prompt.id}-{i}-after-vton.jpg")

        return results

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
            flat_lay = 'flat lay' in tune.title.lower()
            cfg_scale = prompt.vton_cfg_scale or 2

            print(f"Running vton for {tune.name=} {category=} {flat_lay=} {cfg_scale=}")

            # Step 1: Run the model
            base_url = FASHN_BASE_URL_NIGHTLY if prompt.vton_hires else FASHN_BASE_URL_V1
            response = session.post(f'{base_url}/run', json={
                'model_image': "data:image/png;base64, " + pil2base64(image),
                'garment_image': tune.face_swap_images[0],
                'category': category,
                'guidance_scale': cfg_scale,
                'flat_lay': flat_lay,
                'nsfw_filter': False,
                'restore_clothes': category != 'one-pieces',
            }, headers=HEADERS)

            response_data = response.json()
            if 'id' not in response_data:
                print(f"Failed to start VTON process: {response_data.get('error', 'Unknown error')}")
                return image

            process_id = response_data['id']

            # Step 2: Poll for the status
            status_url = f'{base_url}/status/{process_id}'
            start_time = time.time()
            for _ in range(120):
                status_response = session.get(status_url, headers=HEADERS)
                status_data = status_response.json()

                if 'status' not in status_data:
                    print(f"VTON status response missing status: {status_data}")
                    continue
                if status_data['status'] in ['completed', 'failed', 'canceled']:
                    break
                time.sleep(0.5)

            if status_data.get('status') != 'completed' or 'output' not in status_data:
                print(f"Failed to get response from VTON: {status_data.get('error', 'Unknown error')}")
                return image

            # Step 3: Fetch the output
            image_url = status_data['output'][0]
            print(f"Successfully completed VTON. Fetching image from {image_url}. Time={time.time() - start_time:.2f}s")
            image = load_image(image_url)

        return image
