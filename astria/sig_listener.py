import os
import signal

if os.environ.get('MOCK_SERVER'):
    from astria_mock_server import report_infer_job_failure, report_tune_job_failure
else:
    from astria_server import report_infer_job_failure, report_tune_job_failure

# Global flag to signal termination
global terminate
terminate = None

# Global variable to store the currently inferred tune
global current_tune
current_tune = None

global current_train_tune
current_train_tune = None

class TerminateException(Exception):
    pass

def handle_sigterm(signum, frame):
    global terminate
    print(f"SIGTERM signum={signum} received, gracefully shutting down... {current_tune=}")
    terminate = signum

    if current_train_tune:
        print(f"Reporting failure for tune_id={current_train_tune.id}")
        report_tune_job_failure(current_train_tune, "TerminateException")
    # Report failure for all prompts that have not been trained yet
    if current_tune:
        for prompt in current_tune.prompts:
            if prompt.trained_at is None or True:
                report_infer_job_failure(prompt, "TerminateException")

def is_terminated():
    global terminate
    return terminate

def set_current_infer_tune(tune):
    global current_tune
    current_tune = tune

def set_current_train_tune(tune):
    global current_train_tune
    current_train_tune = tune

signal.signal(signal.SIGTERM, handle_sigterm)
if not os.environ.get('DEBUG', None):
    signal.signal(signal.SIGINT, handle_sigterm)
