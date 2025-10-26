import os
import requests
import threading
import time
from io import BytesIO
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from exceptions import ApplicativeApiError
from image_utils import pil2base64_datauri
from vertex_api import gemini_text


class FalClient:
    API_KEY = os.environ.get('FAL_API_KEY')
    HEADERS = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Key {API_KEY}"
    }

    @staticmethod
    def client():
        """Create a configured requests session with retry strategy"""
        if FalClient.API_KEY is None:
            raise ValueError("FAL_API_KEY environment variable not set")

        session = requests.Session()

        # Configure retry strategy: max 3 retries, retry on 504 and 502
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[504, 502],
        )

        # Create adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @staticmethod
    def reve(prompt):
        tunes_images = []
        for tune in prompt.tunes:
            if tune.face_swap_images:
                tunes_images.extend(tune.face_swap_images[:3])

        if prompt.input_image:
            tunes_images.append(prompt.input_image)
        tunes_images = [pil2base64_datauri(i) if isinstance(i, Image.Image) else i for i in tunes_images]

        # Determine which endpoint to use based on image count
        if not tunes_images:
            url = 'https://queue.fal.run/fal-ai/reve/text-to-image'
            payload = {
                "prompt": gemini_text(prompt),
            }
        elif len(tunes_images)==1 and prompt.input_image:
            url = 'https://queue.fal.run/fal-ai/reve/edit'
            payload = {
                "prompt": gemini_text(prompt),
                "image_url": tunes_images[0],
            }
        else:
            url = 'https://queue.fal.run/fal-ai/reve/remix'
            payload = {
                "prompt": gemini_text(prompt),
                "image_urls": tunes_images
            }

        images = FalClient.generic_edit(prompt, url, payload)
        return images

    @staticmethod
    def generic_edit(prompt, url, payload):
        num_images = prompt.num_images or 1

        threads = []
        results = [None] * num_images
        errors = []

        def make_request(i):
            try:
                images = FalClient.generic_image_one(prompt, i, url, payload)
                results[i] = images
            except Exception as e:
                print(f"generic_image_one failed for idx={i}: {type(e).__name__}: {str(e)}")
                errors.append(e)
                results[i] = None

        # Create and start threads
        for i in range(num_images):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        successful = [r for r in results if r is not None]
        if not successful:
            error = errors[0] if errors else Exception("All FAL image requests failed")
            # Try to extract detailed error message
            if hasattr(error, 'response') and error.response is not None:
                try:
                    body = error.response.json()
                    if isinstance(body, dict) and 'detail' in body:
                        detail = body['detail']
                        if isinstance(detail, list) and len(detail) > 0 and 'msg' in detail[0]:
                            raise ApplicativeApiError(detail[0]['msg'])
                except:
                    pass
            raise error

        # Flatten all images from all successful requests
        all_images = []
        for result_images in successful:
            all_images.extend(result_images)

        return all_images

    @staticmethod
    def generic_image_one(prompt, image_index, url, payload):
        # Create session and submit request
        session = FalClient.client()
        response = session.post(url, json=payload, headers=FalClient.HEADERS, timeout=600)
        response.raise_for_status()
        body = response.json()

        # Get status URL for polling
        status_url = body.get('status_url')
        if not status_url:
            raise Exception(f"No status_url in response: {body}")

        timeout = time.time() + 600  # 10 minutes timeout
        while True:
            if time.time() > timeout:
                raise Exception("FAL API request timed out")

            status_response = session.get(status_url, headers=FalClient.HEADERS)
            status_response.raise_for_status()
            status_body = status_response.json()

            status = status_body.get('status')
            if status == 'COMPLETED':
                break
            elif status in ['FAILED', 'CANCELLED']:
                error_msg = status_body.get('error', 'Unknown error')
                raise Exception(f"FAL API request {status}: {error_msg}")

            # Wait before next poll
            if os.environ.get('RAILS_ENV') != 'test':
                time.sleep(1)

        # Get final result
        response_url = status_body.get('response_url')
        if not response_url:
            raise Exception(f"No response_url in status: {status_body}")

        result_response = session.get(response_url, headers=FalClient.HEADERS)
        result_response.raise_for_status()
        result_body = result_response.json()

        # Extract images
        images_data = result_body.get('images')
        if not images_data:
            description = result_body.get('description', 'No description')
            raise Exception(f"FAL processing failed: no images - {description}")

        # Download and convert images to PIL format
        pil_images = []
        for image_hash in images_data:
            image_url = image_hash.get('url')
            if not image_url:
                continue

            # Download image
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()

            # Convert to PIL Image
            image_data = img_response.content
            io_obj = BytesIO(image_data)
            pil_image = Image.open(io_obj)
            pil_images.append(pil_image)

        return pil_images
