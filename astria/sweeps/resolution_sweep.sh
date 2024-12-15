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
export FOLDER_SUFFIX="sweepresolution"
export VALIDATION_STEPS="50"

TRACKER_NAME="res=512" TRAIN_RESOLUTION="512" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=576" TRAIN_RESOLUTION="576" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=640" TRAIN_RESOLUTION="640" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=704" TRAIN_RESOLUTION="704" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=768" TRAIN_RESOLUTION="768" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=832" TRAIN_RESOLUTION="832" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=896" TRAIN_RESOLUTION="896" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=960" TRAIN_RESOLUTION="960" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="res=1024" TRAIN_RESOLUTION="1024" python3 astria/train.py "$RUN_ID"
