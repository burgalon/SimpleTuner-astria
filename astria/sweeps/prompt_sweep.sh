#!/bin/bash

# Check if RUN_ID and TRACKER_PROJECT are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Both RUN_ID and TRACKER_PROJECT are required."
  echo "Usage: $0 <RUN_ID> <TRACKER_PROJECT>"
  exit 1
fi

# Assign the provided arguments to variables
export RUN_ID=$1
export TRACKER_PROJECT=$2

export TRAIN_STEPS=301
export DEBUG=1
export CAPTION_DROPOUT_PROBABILITY="0"
export MOCK_SERVER=1
export TR_REPORT_TO="wandb"
export PREPROCESSING="2"
export FOLDER_SUFFIX="sweepprompt"
export VALIDATION_STEPS="50"

INSTANCE_PROMPT="!TOKEN, a confident and professional individual with a calm demeanor." TRACKER_NAME="prompt=1" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="!TOKEN" TRACKER_NAME="prompt=2" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="A man named !TOKEN." TRACKER_NAME="prompt=3" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="!TOKEN has a smooth, bald head with a tanned complexion. His face is oval-shaped, with prominent cheekbones and a strong, angular jawline. He has dark, deep-set eyes beneath slightly arched eyebrows, a straight nose, and thin lips that are closed in a neutral expression. His skin has a few subtle wrinkles, particularly around the eyes and forehead, suggesting maturity. His ears are well-proportioned and sit close to his head." TRACKER_NAME="prompt=4" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="Photo of !TOKEN" TRACKER_NAME="prompt=5" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="!TOKEN, a bald man with a tanned complexion" TRACKER_NAME="prompt=6" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="A portrait style photograph of !TOKEN" TRACKER_NAME="prompt=7" python3 astria/train.py "$RUN_ID"
INSTANCE_PROMPT="!TOKEN, a man" TRACKER_NAME="prompt=8" python3 astria/train.py "$RUN_ID"
