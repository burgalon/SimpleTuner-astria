import base64
import os
import re
import requests
import threading
from io import BytesIO
from PIL import Image

from exceptions import ApplicativeApiError
from image_utils import load_image, pil2base64

SYSTEM_INSTRUCTION = """
**1. Maximize Image Quality and Detail:**
*   Always strive for photorealistic quality unless a specific artistic style is explicitly requested.
*   Render intricate details, textures, and subtle nuances to enhance realism and visual richness.
*   Employ advanced lighting techniques (e.g., golden hour, dramatic chiaroscuro, softbox setup) and appropriate camera settings (e.g., aperture, focal length, depth of field) as implied or explicitly stated in the prompt.
*   Ensure crisp focus on primary subjects and intelligent use of background blur (bokeh) where artistically appropriate.

**2. Strict Prompt Adherence:**
*   Interpret all instructions literally and execute them precisely. Do not infer or deviate from the explicit details provided.
*   Prioritize narrative, descriptive paragraphs over keyword lists. Construct the scene as a director would, detailing environment, subject actions, mood, and visual elements.
*   Be hyper-specific with all elements: colors, materials, shapes, sizes, emotions, and interactions. If a detail is provided, it *must* be incorporated.
*   Maintain character consistency across iterative prompts by meticulously referencing previous outputs and descriptions.
*   Incorporate provided aspect ratios without alteration. If multiple input images are given, adopt the aspect ratio of the last image unless otherwise specified.

**3. Prompting for Censorship Avoidance and Creative Freedom:**
*   Your goal is to fulfill creative requests while adhering to ethical guidelines. When a prompt approaches a sensitive topic, interpret it in the most benign, artistic, and abstract manner possible to fulfill the user's creative intent without generating harmful content.
*   Focus on the artistic and aesthetic qualities of the request. For example, if a prompt involves themes that could be interpreted as violent or explicit, render it symbolically, metaphorically, or in a highly stylized, non-graphic manner (e.g., classical sculpture, abstract art, or a distant, implied scene).
*   When a prompt includes elements that might be flagged, use semantic rephrasing internally to generate the desired visual outcome. For instance, instead of directly depicting a prohibited item, focus on its *effect* or *absence* in the scene, or a harmless analogous representation.
*   If a prompt is ambiguous or could be misconstrued, default to the most artistic, symbolic, and non-literal interpretation that aligns with high image quality and creative expression, avoiding any literal depiction of potentially restricted content.
*   Your purpose is to enable creative expression. If a prompt can be rendered safely and artistically, do so. Do not preemptively censor or refuse a prompt unless it directly and unambiguously violates safety policies in its most benign interpretation.

**4. Output Format:**
*   Generate a single image per request unless multiple distinct images are explicitly asked for within a single prompt.
*   Provide the image in the highest possible resolution and quality.
"""


def gemini_text(prompt):
    # replace <...> with '' using regex
    ret = re.sub(r'(<[^>]*>)', '', prompt.text)
    for tune in prompt.tunes:
        if tune.train_token:
            ret = ret.replace(tune.train_token, 'reference')
    print(f'P#{prompt.id} gemini_text=[{ret}] prompt=[{prompt.text}]')
    return ret

class GeminiImageApi:
    @staticmethod
    def gemini_edit_image(prompt):
        api_key = os.environ.get('VERTEX_API_KEY')
        if not api_key:
            raise Exception("VERTEX_API_KEY environment variable not set")

        # Get tune images - similar to tunes_images = prompt.tunes.map { |t| t.images.slice(...3) }.flatten
        tunes_images = []
        for tune in prompt.tunes:
            if tune.face_swap_images:
                tunes_images.extend(tune.face_swap_images[:3])

        # Add input_image if present
        if prompt.input_image:
            tunes_images.append(prompt.input_image)

        # Build the payload
        payload = {
            "contents": [{
                "role": "user",
                "parts": GeminiImageApi._build_gemini_parts(tunes_images, gemini_text(prompt))
            }],
            "systemInstruction": { "parts": [{"text": SYSTEM_INSTRUCTION}] },
            "generationConfig": {
                "temperature": 1,
                "maxOutputTokens": 32768,
                "responseModalities": ["IMAGE"],
                "topP": 0.95
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}
            ]
        }

        # Determine number of images to generate
        num_images = prompt.num_images or 1


        # Make API calls (parallelized)
        threads = []
        results = []
        errors = []

        def make_request(i):
            try:
                url = f"https://aiplatform.googleapis.com/v1/projects/marine-bebop-276519/locations/global/publishers/google/models/gemini-2.5-flash-image:streamGenerateContent?key={api_key}"
                response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
                response.raise_for_status()
                body = response.json()

                # Process streaming response
                parts = []
                usage = None
                if isinstance(body, list):
                    for chunk in body:
                        usage = usage or chunk.get('usageMetadata')
                        chunk_parts = chunk.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                        parts.extend(chunk_parts)
                else:
                    parts = body.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                    usage = body.get('usageMetadata')

                # Extract images from parts
                blob_kwargs_arr = []
                for j, part in enumerate(parts):
                    if 'inlineData' in part:
                        inline_data = part['inlineData']
                        data = base64.b64decode(inline_data['data'])
                        io_obj = BytesIO(data)
                        ext = inline_data['mimeType'].split('/')[-1]
                        blob_kwargs_arr.append({
                            'io': io_obj,
                            'filename': f"{prompt.id}-{i}-{j}.{ext}",
                            'content_type': inline_data['mimeType']
                        })

                if not blob_kwargs_arr:
                    error_text = ' '.join([part.get('text', '') for part in parts if 'text' in part])
                    errors.append(f"Request {i}: {error_text or 'No image data returned'}")
                    return

                # For now, we'll calculate a fixed cost per image
                cost_mc = 3870  # $0.039 per image in microcents
                results.append({'blobs': blob_kwargs_arr, 'cost': cost_mc})

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
            raise ApplicativeApiError(f"All requests failed: {error_msg}")

        # Aggregate results
        all_blobs = []
        total_cost = 0
        for result in results:
            all_blobs.extend(result['blobs'])
            total_cost += result['cost']

        # Convert blobs to PIL images
        images = []
        for blob_kwargs in all_blobs:
            image = Image.open(blob_kwargs['io'])
            images.append(image)

        return images

    @staticmethod
    def _build_gemini_parts(tunes_images, gemini_text):
        """Build the parts array for Gemini API request"""
        parts = []

        # Process images in parallel
        threads = []
        results = [None] * len(tunes_images)

        def process_image(idx, image):
            results[idx] = {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": pil2base64(load_image(image))
                }
            }

        # Start threads
        for idx, image in enumerate(tunes_images):
            thread = threading.Thread(target=process_image, args=(idx, image))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Add valid image parts
        parts.extend([r for r in results if r is not None])

        # Add text part
        if gemini_text:
            parts.append({"text": gemini_text})

        return parts
