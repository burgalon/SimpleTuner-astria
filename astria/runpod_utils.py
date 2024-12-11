import os

from astria_utils import run

def kill_pod():
    run(['runpodctl', 'remove', 'pod', os.environ['RUNPOD_POD_ID']])

