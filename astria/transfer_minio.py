import time
import os
from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager, Timeout
from urllib3.exceptions import ReadTimeoutError

http_client = PoolManager(
    timeout=Timeout(connect=5, read=15),
    # You can also add other urllib3 settings here, like retries
    # retries=Retry(total=5, redirect=2, connect=3),
)

if not os.environ.get('R2_ENDPOINT_URL_S3'):
    os.environ.setdefault('R2_ENDPOINT_URL_S3', "")
if not os.environ.get('R2_ACCESS_KEY_ID'):
    os.environ.setdefault('R2_ACCESS_KEY_ID', "")
if not os.environ.get('R2_SECRET_ACCESS_KEY'):
    os.environ.setdefault('R2_SECRET_ACCESS_KEY', "")

client = Minio(os.environ.get('R2_ENDPOINT_URL_S3', 'http://dummy').split('/')[-1],
               access_key=os.environ.get('R2_ACCESS_KEY_ID', 'dummy'),
               secret_key=os.environ.get('R2_SECRET_ACCESS_KEY', 'dummy'),
               http_client=http_client,
               )

BUCKET = 'sdbooth2-production'
MAX_RETRIES = 5

def upload_minio(file, target):
    start_time = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            client.fput_object(BUCKET, target, file)
            end_time = time.time()
            print(f"Uploaded {target} in {end_time - start_time:.2f} seconds")
            return
        except (S3Error, ReadTimeoutError) as exc:
            print(f"Upload of {target} failed on attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 1
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to upload {target} after {MAX_RETRIES} attempts.")
                raise

def download_minio(from_key, to):
    start_time = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            client.fget_object(BUCKET, from_key, to)
            end_time = time.time()
            print(f"Downloaded {from_key} in {end_time - start_time:.2f} seconds")
            return
        except (S3Error, ReadTimeoutError) as exc:
            print(f"Download of {from_key} failed on attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 1
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to download {from_key} after {MAX_RETRIES} attempts.")
                raise

if __name__ == '__main__':
    upload_minio('/data/models/9.safetensors', 'models/9.safetensors')
    # download_minio('models/2002368.safetensors', '/data/models/2002368.safetensors')
