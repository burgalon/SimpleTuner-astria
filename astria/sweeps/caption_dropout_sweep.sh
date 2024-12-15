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
export MOCK_SERVER=1
export TR_REPORT_TO="wandb"
export INSTANCE_PROMPT="!TOKEN, a confident and professional individual with a calm demeanor."
export PREPROCESSING="2"
export FOLDER_SUFFIX="sweepcapdrop"
export VALIDATION_STEPS="50"

TRACKER_NAME="c_drop=0.0" CAPTION_DROPOUT_PROBABILITY="0.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.1" CAPTION_DROPOUT_PROBABILITY="0.1" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.2" CAPTION_DROPOUT_PROBABILITY="0.2" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.3" CAPTION_DROPOUT_PROBABILITY="0.3" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.4" CAPTION_DROPOUT_PROBABILITY="0.4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.5" CAPTION_DROPOUT_PROBABILITY="0.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.6" CAPTION_DROPOUT_PROBABILITY="0.6" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.7" CAPTION_DROPOUT_PROBABILITY="0.7" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.8" CAPTION_DROPOUT_PROBABILITY="0.8" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=0.9" CAPTION_DROPOUT_PROBABILITY="0.9" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="c_drop=1.0" CAPTION_DROPOUT_PROBABILITY="1.0" python3 astria/train.py "$RUN_ID"