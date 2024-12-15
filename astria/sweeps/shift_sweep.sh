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
export FOLDER_SUFFIX="sweepshift"
export VALIDATION_STEPS="50"

TRACKER_NAME="shift=0.0" FLUX_SCHEDULE_SHIFT="0.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=0.5" FLUX_SCHEDULE_SHIFT="0.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=1.0" FLUX_SCHEDULE_SHIFT="1.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=1.5" FLUX_SCHEDULE_SHIFT="1.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=2.0" FLUX_SCHEDULE_SHIFT="2.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=2.5" FLUX_SCHEDULE_SHIFT="2.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=-0.5" FLUX_SCHEDULE_SHIFT="-0.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=-1.0" FLUX_SCHEDULE_SHIFT="-1.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=-1.5" FLUX_SCHEDULE_SHIFT="-1.5" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=-2.0" FLUX_SCHEDULE_SHIFT="-2.0" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="shift=-2.5" FLUX_SCHEDULE_SHIFT="-2.5" python3 astria/train.py "$RUN_ID"