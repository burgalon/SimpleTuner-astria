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
export FOLDER_SUFFIX="sweeptoken"
export VALIDATION_STEPS="50"

TRAIN_TOKEN="ohwx" TRACKER_NAME="token=ohwx" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="Donnie T. Googe" TRACKER_NAME="token=googe" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="ohwnsixiolwevo" TRACKER_NAME="token=ohwnsixiolwevo" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="4727341682" TRACKER_NAME="token=4727341682" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="James Mason" TRACKER_NAME="token=mason" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="ohwx94827462" TRACKER_NAME="token=ohwx94827462" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="General Tom Eustice" TRACKER_NAME="token=general" python3 astria/train.py "$RUN_ID"
TRAIN_TOKEN="Decorated War Veteran Jason Littlefield" TRACKER_NAME="token=vet" python3 astria/train.py "$RUN_ID"
