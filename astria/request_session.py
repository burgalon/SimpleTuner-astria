import json

import requests

def handle_response(r, *args, **kwargs):
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        # Print response text which is assumed to be the error message
        print(f"Error response body: {r.text}")
        # try to parse json from body
        try:
            body = json.loads(r.text)
            if 'error' in body:
                # error_description = body['errors']
                raise requests.exceptions.HTTPError(body['error'])
        except json.JSONDecodeError:
            pass
        # Re-raise the exception after handling
        raise


session = requests.Session()
retries = requests.packages.urllib3.util.retry.Retry(total=10, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 403, 401], raise_on_status=True)
session.mount('', requests.adapters.HTTPAdapter(max_retries=retries))
session.hooks = {
    'response': handle_response,
}
