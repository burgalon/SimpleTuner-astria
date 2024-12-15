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
export FOLDER_SUFFIX="sweepbsz"
export VALIDATION_STEPS="50"

TRACKER_NAME="bsz=1" TRAIN_BATCH="1" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="bsz=2" TRAIN_BATCH="2" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="bsz=3" TRAIN_BATCH="3" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="bsz=4" TRAIN_BATCH="4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="bsz=5" TRAIN_BATCH="5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="bsz=6" TRAIN_BATCH="6" python3 astria/train.py "$RUN_ID"
