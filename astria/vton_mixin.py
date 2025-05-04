import os
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import rollbar
from PIL import Image
from astria_utils import JsonObj, MODELS_DIR
if os.environ.get('MOCK_SERVER') or os.environ.get('DEBUG') == 'test':
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

HEADERS = {"Authorization": f"Key {FASHN_API_KEY}", "Content-Type": "application/json"}
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

        # if os.environ.get('DEBUG'):
        #     for i, image in enumerate(results):
        #         image.save(MODELS_DIR + f"/{prompt.id}-{i}-after-vton.jpg")

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
            cfg_scale = prompt.vton_cfg_scale or 2

            garment_photo_type = 'model' if 'model' in tune.title else 'flat-lay' if 'flat-lay' in tune.title else 'auto'
            print(f"Running vton for {tune.name=} {category=} {cfg_scale=} {garment_photo_type=}")

            # Step 1: Run the model
            base_url = 'https://queue.fal.run/fal-ai/fashn/tryon/v1.5'
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
                if result_data['detail'] and isinstance(result_data['detail'], list):
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
