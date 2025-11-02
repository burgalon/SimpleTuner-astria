from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from astria_server import DOMAIN
import time
import base64
import hashlib
import io
from PIL import Image

from image_utils import s3_session


def process_and_upload_image(image, content_type, id, i_image):
    if isinstance(image, bytes):
        # this is a video - just calculate the checksum
        md5 = base64.b64encode(hashlib.md5(image).digest()).decode()
        byte_size = len(image)
        img_byte_arr = image
        filename = f"{id}-{i_image}.mp4"
    else:
        # Convert PIL Image to byte stream
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG" if content_type == "image/png" else "JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # Calculate byte size and checksum
        byte_size = len(img_byte_arr)
        md5 = base64.b64encode(hashlib.md5(img_byte_arr).digest()).decode()
        # Create JSON data for the request
        filename = f"{id}-{i_image}." + ("png" if content_type == "image/png" else "jpg")

    json_data = {"blob": {"filename": filename, "byte_size": byte_size,
                          "metadata": {"width": image.width, "height": image.height} if isinstance(image, Image.Image) else {},
                          "content_type": content_type, "checksum": md5}}

    # Send request to get the upload URL
    response = s3_session.post(f"{DOMAIN}api/direct_uploads", json=json_data, headers={"Content-Type": "application/json", "Authorization": "Bearer sd_bor2YahMCYBDSdXqN1wDe1A4cnBPaJ"}, timeout=(3,4))
    response.raise_for_status()
    response_json = response.json()
    upload_url = response_json["direct_upload"]["url"]
    upload_headers = response_json["direct_upload"]["headers"]

    # remove signature query string from upload_url
    print(f"Uploading {filename} {byte_size} bytes md5={md5} upload_url={upload_url.split('?')[0]}")

    # Upload the image
    upload_response = s3_session.put(upload_url, data=img_byte_arr, headers=upload_headers, timeout=(3,4))
    upload_response.raise_for_status()

    return ("prompt[images][]", response_json['signed_id'])

def send_to_server(images: [Image.Image, bytes], id: int, content_types=None):
    content_types = content_types or ["image/jpeg"] * len(images)

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        futures = []
        for i_image, (content_type, image) in enumerate(zip(content_types, images)):
            futures.append(executor.submit(process_and_upload_image, image, content_type, id, i_image))

        done, not_done = wait(futures, return_when=ALL_COMPLETED)
        upload_files = [future.result() for future in futures]

    print(f"Sending images id={id} {(time.time() - start_time):.2f} seconds")

    # Send to Rails server
    start_time = time.time()
    s3_session.post(f"{DOMAIN}prompts/{id}/done", data=upload_files)
    # if more than 2 seconds, log it
    if time.time() - start_time > 0:
        print(f"Sending prompt done id={id} {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    print(f"Testing send to server")
    send_to_server([Image.new('RGB', (100, 100))], 1, ["image/jpeg"])

