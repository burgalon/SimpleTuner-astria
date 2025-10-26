import base64
import os
import re
import requests
import threading
from io import BytesIO
from PIL import Image
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from image_utils import load_image
from vertex_api import gemini_text
from exceptions import ApplicativeApiError

MAX_DIMENSION = 4096

def resize_image_if_needed(image_data, content_type, max_size_mb=10):
    """Helper function to resize image if it's larger than max_size_mb MB or exceeds MAX_DIMENSION"""
    max_size = max_size_mb * 1024 * 1024

    try:
        # Create PIL Image from data
        image = Image.open(BytesIO(image_data))
        original_width, original_height = image.size
        max_pixels = MAX_DIMENSION * MAX_DIMENSION

        # Check if we need to resize based on dimensions, file size, or format
        needs_dimension_resize = original_width * original_height > max_pixels
        needs_file_size_resize = len(image_data) > max_size
        needs_format_resize = content_type.lower() not in ["image/jpeg", "image/png"]

        # If no resizing needed, return original
        if not needs_dimension_resize and not needs_file_size_resize and not needs_format_resize:
            return image_data, content_type

        # Convert to RGB if necessary and prepare for JPEG conversion
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Resize dimensions if needed
        if needs_dimension_resize:
            # Calculate scaling factor to fit within MAX_DIMENSIONxMAX_DIMENSION while preserving aspect ratio
            k = math.sqrt(max_pixels / (original_width * original_height))
            new_width = int(original_width * k - 1)
            new_height = int(original_height * k - 1)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save as JPEG with quality 85 first
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=85)
        resized_data = output_buffer.getvalue()

        if len(resized_data) <= max_size:
            return resized_data, "image/jpeg"

        # If still too large after dimension resizing, reduce quality
        quality = 75
        while quality >= 20:
            output_buffer = BytesIO()
            image.save(output_buffer, format='JPEG', quality=quality)
            resized_data = output_buffer.getvalue()

            # Check if size is now acceptable
            if len(resized_data) <= max_size:
                return resized_data, "image/jpeg"

            # Reduce quality for next iteration
            quality -= 10

        # If we still haven't achieved the target size, return the last attempt
        print(f"Warning: Could not resize image below {max_size_mb}MB, final size: {len(resized_data)} bytes")
        return resized_data, "image/jpeg"

    except Exception as e:
        print(f"Error: Failed to resize image: {e}")
        # Return original data if resizing fails
        return image_data, content_type


class ByteplusClient:
    API_KEY = os.environ.get('BYTEPLUS_API_KEY')
    HEADERS = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {API_KEY}"
    }

    @staticmethod
    def client(context_prefix=""):
        """Create a configured requests session matching the Ruby Faraday client"""
        if ByteplusClient.API_KEY is None:
            raise ValueError("BYTEPLUS_API_KEY environment variable not set")
        session = requests.Session()

        # Configure retry strategy: max 5 retries, interval 7 seconds, retry on 500 and 429
        retry_strategy = Retry(
            total=5,
            backoff_factor=7,  # interval multiplier
            status_forcelist=[500, 429],  # retry on these status codes
        )

        # Create adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @staticmethod
    def seedream_edit(prompt):
        """Generate images using Byteplus Seedream API"""
        # Get tune images - similar to tunes_images = prompt.tunes.map { |t| t.images.slice(...3) }.flatten
        tunes_images = []
        for tune in prompt.tunes:
            if tune.face_swap_images:
                tunes_images.extend(tune.face_swap_images[:3])

        # Add input_image if present
        if prompt.input_image:
            # Important to use insert so that we keep aspect ratio according to the input_image
            # Which is usually the reference image
            tunes_images.insert(0, prompt.input_image)

        # Convert all images to PIL.Image.Image objects
        tunes_images = [load_image(img) for img in tunes_images]

        # Determine size
        if prompt.w and prompt.h:
            size = f"{prompt.w}x{prompt.h}"
        elif prompt.aspect_ratio:
            # max is 16777216
            dims = {
                '1:1': [2048, 2048],
                '4:3': [2304, 1728],
                '3:4': [1728, 2304],
                '16:9': [2560, 1440],
                '9:16': [1440, 2560],
                '3:2': [2496, 1664],
                '2:3': [1664, 2496],
                '21:9': [3024, 1296]
            }.get(prompt.aspect_ratio, [4096, 4096])
            size = f"{dims[0]}x{dims[1]}"
        else:
            first_image = tunes_images[0] if tunes_images else None
            if first_image:
                w, h = first_image.size
                k = math.sqrt(MAX_DIMENSION * MAX_DIMENSION / (w * h))
                w = math.floor(w * k)
                h = math.floor(h * k)
                size = f"{w}x{h}"
            else:
                size = "4096x4096"

        # Process images and convert to base64
        base64_images = []
        for image in tunes_images:
            # Convert PIL image to bytes
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            content_type = "image/png"

            # Resize image if it's larger than 10MB
            resized_data, final_content_type = resize_image_if_needed(image_data, content_type)

            base64_images.append(f"data:{final_content_type};base64,{base64.b64encode(resized_data).decode('utf-8')}")

        # Determine number of images to generate
        num_images = prompt.num_images or 1

        # Make API calls (parallelized)
        threads = []
        results = []
        errors = []

        def make_request(i):
            # Build payload with thread-specific seed
            payload = {
                # No filter model
                "model": "ep-20250912183030-gx7q5",
                "prompt": gemini_text(prompt),
                "image": base64_images,
                "sequential_image_generation": "disabled",
                "response_format": "b64_json",
                "size": size,
                "seed": (prompt.seed or 42) + i,
                "stream": False,
                "watermark": False
            }
            try:
                session = ByteplusClient.client()
                url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
                response = session.post(url, json=payload, headers=ByteplusClient.HEADERS, timeout=240)
                response.raise_for_status()
                body = response.json()

                if os.environ.get('DEBUG'):
                    print(f"Debug: Byteplus response {i}: {re.sub(r'[A-Za-z0-9+/=]{32,}', '[BASE64_REDACTED]', str(body))}")

                # Extract images from response
                data = body.get("data", [])
                images = []
                for j, item in enumerate(data):
                    if 'b64_json' in item:
                        image_data = base64.b64decode(item['b64_json'])
                        io_obj = BytesIO(image_data)
                        pil_image = Image.open(io_obj)
                        images.append(pil_image)
                    elif 'error' in item:
                        error_msg = item['error'].get('message', 'Unknown error')
                        errors.append(f"Request {i}, item {j}: {error_msg}")

                if not images:
                    errors.append(f"Request {i}: No valid images returned")
                    return

                results.append(images)

            except requests.exceptions.HTTPError as e:
                # Log validation errors (status codes 400-599 except 400 and 500)
                status_code = e.response.status_code
                if 400 <= status_code <= 599 and status_code not in [400, 500]:
                    print(f"Byteplus API error (status {status_code}): {str(e)}")

                # Handle 400 Bad Request with specific error format
                if status_code == 400:
                    try:
                        error_body = e.response.json()
                        if 'error' in error_body and 'message' in error_body['error']:
                            raise ApplicativeApiError(error_body['error']['message'])
                    except:
                        pass
                errors.append(f"Request {i}: {str(e)}")
            except Exception as e:
                errors.append(f"Request {i}: {str(e)}")

        # Create threads for parallel requests
        for i in range(num_images):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        if not results:
            error_msg = "; ".join(errors)
            raise ApplicativeApiError(error_msg or "All requests failed")

        # Aggregate results - flatten all images from all requests
        all_images = []
        for result_images in results:
            all_images.extend(result_images)

        return all_images
