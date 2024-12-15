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
export INSTANCE_PROMPT="!TOKEN, a confident and professional individual with a calm demeanor."
export PREPROCESSING="2"
export FOLDER_SUFFIX="sweeplr"
export VALIDATION_STEPS="50"

TRACKER_NAME="lr=1e-4" LEARNING_RATE="1e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=2e-4" LEARNING_RATE="2e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=3e-4" LEARNING_RATE="3e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=4e-4" LEARNING_RATE="4e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=5e-4" LEARNING_RATE="5e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=6e-4" LEARNING_RATE="6e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=7e-4" LEARNING_RATE="7e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=8e-4" LEARNING_RATE="8e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=9e-4" LEARNING_RATE="9e-4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="lr=1e-3" LEARNING_RATE="1e-3" python3 astria/train.py "$RUN_ID"
