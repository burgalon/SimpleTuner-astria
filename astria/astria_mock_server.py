import json
from PIL import Image

from astria_utils import JsonObj

FASHN_API_KEY = 'TEST'
DB_JSON = 'astria_tests/fixtures/db.json'

def report_tune_job_failure(tune: JsonObj, fail_message: str):
    pass

def report_infer_job_failure(prompt: JsonObj, trace_str: str):
    pass

def request_tune_job_from_server(id = None):
    # read db from json file
    id = int(id)
    with open(DB_JSON, 'r') as f:
        db = json.load(f, object_hook=lambda d: JsonObj(**d))
    if id is None:
        raise ValueError("Mock server should work with id")
    tune = next(t for t in db if t.id == id)
    return tune

def request_infer_job_from_server(tune_id: str = None, is_video: bool = False, id: str = None):
    # read db from json file
    with open(DB_JSON, 'r') as f:
        db = json.load(f, object_hook=lambda d: JsonObj(**d))
    if id is None:
        raise ValueError("Mock server should work with id")
    id = int(id)
    for tune in db:
        prompt = next(p for p in db.prompts if p.id == id)
        tune.prompts = [prompt]
        return tune
    raise ValueError(f"Invalid id {id}")
def server_tune_done(tune: JsonObj):
    pass

def send_to_server(images: [Image.Image], id: int, content_types=None):
    pass
